package com.lite.holistic_tracking;

import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.os.Build;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;

import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

import android.widget.TextView;
import android.widget.Toast;

import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.glutil.EglManager;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.flex.FlexDelegate;
import org.tensorflow.lite.support.common.FileUtil;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class holistic_activity extends AppCompatActivity {


    private static final String TAG = "MainActivity";

    // Flips the camera-preview frames vertically by default, before sending them into FrameProcessor
    // to be processed in a MediaPipe graph, and flips the processed frames back when they are
    // displayed. This maybe needed because OpenGL represents images assuming the image origin is at
    // the bottom-left corner, whereas MediaPipe in general assumes the image origin is at the
    // top-left corner.
    // NOTE: use "flipFramesVertically" in manifest metadata to override this behavior.
    private static final boolean FLIP_FRAMES_VERTICALLY = true;

    static {
        // Load all native libraries needed by the app.
        System.loadLibrary("mediapipe_jni");
        try {
            System.loadLibrary("opencv_java3");
        } catch (UnsatisfiedLinkError e) {
            // Some example apps (e.g. template matching) require OpenCV 4.
            System.loadLibrary("opencv_java4");
        }
    }

    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    protected FrameProcessor processor_pose_hand;
    private static final String INPUT_NUM_HANDS_SIDE_PACKET_NAME = "num_hands";
    private static final String INPUT_MODEL_COMPLEXITY = "model_complexity";
    private static final int NUM_HANDS = 2; // Set the number of hands you want to detect


    // Handles camera access via the {@link CameraX} Jetpack support library.
    protected CameraXPreviewHelper cameraHelper;

    String lastPredictedGesture;

    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private SurfaceTexture previewFrameTexture;
    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private SurfaceView previewDisplayView;

    // Creates and manages an {@link EGLContext}.
    private EglManager eglManager;
    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private ExternalTextureConverter converter;

    // ApplicationInfo for retrieving metadata defined in the manifest.
    private ApplicationInfo applicationInfo;

    // UI Components
    private Button rotateButton, startPauseButton, enableTtsButton, infoButton;

    private TextView translatedText, fps_meter;

    private boolean isCameraStarted = false, isTtsEnabled = false, isTtsReady = false;
    private CameraHelper.CameraFacing currentCameraFacing = CameraHelper.CameraFacing.BACK;


    private Interpreter tflite;

    private TextToSpeech mTTS;


    Map<Long, LandmarkData> landmarkMap = new HashMap<>();

    Deque<float[]> landmarksQueue = new LinkedList<>();
    private Deque<String> lastPredictions = new ArrayDeque<>(4);

    private long startTime = System.currentTimeMillis();
    private int frameCount = 0;

    String[] gestures = {
            "beliau", "panas", "boleh", "masa", "setiap", "mahu", "membaca", "kerja", "berfikir", "tolong"
    };



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //TTS functionality
        mTTS = new TextToSpeech(this, status -> {
            if (status==TextToSpeech.SUCCESS){
                int result = mTTS.setLanguage(Locale.ENGLISH);
                if (result == TextToSpeech.LANG_MISSING_DATA
                        || result == TextToSpeech.LANG_NOT_SUPPORTED){
                    Log.d(TAG, "TTS ERROR");
                } else {
                    enableTtsButton.setEnabled(true);
                }
            }else{
                Log.d(TAG, "TTS NOT Success");
            }

        });
        setContentView(R.layout.activity_holistic_activity);

        PermissionHelper.checkAndRequestCameraPermissions(this);

        try {
            applicationInfo =
                    getPackageManager().getApplicationInfo(getPackageName(), PackageManager.GET_META_DATA);
        } catch (PackageManager.NameNotFoundException e) {
            Log.e(TAG, "Cannot find application info: " + e);
        }


        // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
        // binary graphs.
        initializeMediapipe();

        // Initialize UI components
        rotateButton = findViewById(R.id.rotate);
        startPauseButton = findViewById(R.id.start_stop);
        enableTtsButton = findViewById(R.id.speaker);
        translatedText = findViewById(R.id.sign_text);
        fps_meter = findViewById(R.id.fps_meter);
        infoButton = findViewById(R.id.infoButton);




        // Set up button listeners
        setupButtonListeners();
        initializeInterpreter();




    }

    private void setupButtonListeners() {
        rotateButton.setOnClickListener(v -> toggleCameraFacing());
        startPauseButton.setOnClickListener(v -> toggleCameraState());
        enableTtsButton.setOnClickListener(v -> ttsSpeak());
        infoButton.setOnClickListener(v -> {
            Log.d(TAG, "INFO BUTTON CLICKED");
            InformationFragment infoFragment = new InformationFragment();
            getSupportFragmentManager().beginTransaction()
                    .replace(R.id.fragment_container, infoFragment)
                    .commit();
            findViewById(R.id.fragment_container).setVisibility(View.VISIBLE);
        });


    }

    @Override
    protected void onDestroy() {
        if (mTTS!=null){
            mTTS.stop();
            mTTS.shutdown();
        }
        super.onDestroy();
    }

    private void ttsSpeak(){
        String text = translatedText.getText().toString();
        mTTS.setPitch(1.0f);
        mTTS.setSpeechRate(1.0f);
        mTTS.speak(text, TextToSpeech.QUEUE_FLUSH, null, null);


}
    private void toggleCameraFacing() {
        // Logic to toggle camera facing
        if (currentCameraFacing == CameraHelper.CameraFacing.FRONT) {
            currentCameraFacing = CameraHelper.CameraFacing.BACK;
            rotateButton.setText("Front");
        } else {
            currentCameraFacing = CameraHelper.CameraFacing.FRONT;
            rotateButton.setText("Back");
        }
        restartCamera();
    }

    private void toggleCameraState() {
        if (isCameraStarted) {
            if (converter != null) {
                converter.close();
                converter = null;
                isCameraStarted = false;
                startPauseButton.setText("Start");
                translatedText.setText("");
                Log.d(TAG, "Camera Paused ");
            }
        } else {
            startCamera();
            startPauseButton.setText("Pause");
            Log.d(TAG, "Camera Resumed ");
        }

    }

    private void restartCamera() {
        if (isCameraStarted) {
            if (converter != null) {
                converter.close();
                converter = null;
            }
            startCamera();
        }
    }


    @Override
    protected void onResume() {
        super.onResume();

        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        converter.close();
        if (mTTS!=null){
            mTTS.stop();
            mTTS.shutdown();
        }
    }


    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    protected void onCameraStarted(SurfaceTexture surfaceTexture) {
        previewFrameTexture = surfaceTexture;
        // Make the display view visible to start showing the preview. This triggers the
        // SurfaceHolder.Callback added to (the holder of) previewDisplayView.
        previewDisplayView.setVisibility(View.VISIBLE);
    }

    protected Size cameraTargetResolution() {

        return null; // No preference and let the camera (helper) decide.
    }

    public void startCamera() {
        previewDisplayView = new SurfaceView(this);
        setupPreviewDisplayView();
        converter = new ExternalTextureConverter(eglManager.getContext());
        converter.setFlipY(
                applicationInfo.metaData.getBoolean("flipFramesVertically", FLIP_FRAMES_VERTICALLY));
        converter.setConsumer(processor_pose_hand);
        cameraHelper = new CameraXPreviewHelper();

        cameraHelper.setOnCameraStartedListener(
                surfaceTexture -> {
                    onCameraStarted(surfaceTexture);
                });
        CameraHelper.CameraFacing cameraFacing = currentCameraFacing;

        cameraHelper.startCamera(
                this, cameraFacing, /*surfaceTexture=*/ null, cameraTargetResolution());
        isCameraStarted = true;
    }

    protected Size computeViewSize(int width, int height) {
        return new Size(width, height);
    }

    protected void onPreviewDisplaySurfaceChanged(
            SurfaceHolder holder, int format, int width, int height) {
        // (Re-)Compute the ideal size of the camera-preview display (the area that the
        // camera-preview frames get rendered onto, potentially with scaling and rotation)
        // based on the size of the SurfaceView that contains the display.
        Size viewSize = computeViewSize(width, height);
        Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);
        boolean isCameraRotated = cameraHelper.isCameraRotated();

        // Connect the converter to the camera-preview frames as its input (via
        // previewFrameTexture), and configure the output width and height as the computed
        // display size.
        converter.setSurfaceTextureAndAttachToGLContext(
                previewFrameTexture,
                isCameraRotated ? displaySize.getHeight() : displaySize.getWidth(),
                isCameraRotated ? displaySize.getWidth() : displaySize.getHeight());
    }

    private void setupPreviewDisplayView() {
        previewDisplayView.setVisibility(View.GONE);
        ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
        viewGroup.addView(previewDisplayView);

        previewDisplayView
                .getHolder()
                .addCallback(
                        new SurfaceHolder.Callback() {
                            @Override
                            public void surfaceCreated(SurfaceHolder holder) {
                                processor_pose_hand.getVideoSurfaceOutput().setSurface(holder.getSurface());
                            }

                            @Override
                            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                                onPreviewDisplaySurfaceChanged(holder, format, width, height);
                            }

                            @Override
                            public void surfaceDestroyed(SurfaceHolder holder) {
                                processor_pose_hand.getVideoSurfaceOutput().setSurface(null);
                            }
                        });
    }

    public class LandmarkData {
        float[] poseLandmarks;
        float[] leftHandLandmarks;
        float[] rightHandLandmarks;



        public LandmarkData() {
            // Initialize with zero-filled arrays
            poseLandmarks = createZeroLandmarks(33, 4); // 33 landmarks for pose, each with x, y, z
            leftHandLandmarks = createZeroLandmarks(21, 3); // 21 landmarks for each hand
            rightHandLandmarks = createZeroLandmarks(21, 3);

        }
    }

    private float[] createZeroLandmarks(int count, int var) {
        // Each landmark has x, y, z coordinates. Hence, count * 3.
        return new float[count * var];
    }

    private float[] convertLandmarksToList(NormalizedLandmarkList landmarks) {
        float[] landmarksArray = new float[21 * 3];
        int index = 0;
        for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
            landmarksArray[index++] = landmark.getX();
            landmarksArray[index++] = landmark.getY();
            landmarksArray[index++] = landmark.getZ();
        }
        return landmarksArray;
    }
    private float[] convertLandmarksToListPose(NormalizedLandmarkList landmarks) {
        float[] landmarkList = new float[33 * 4]; // 33 landmarks, each with 4 values (x, y, z, visibility)
        int i = 0;
        for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
            landmarkList[i++] = landmark.getX();
            landmarkList[i++] = landmark.getY();
            landmarkList[i++] = landmark.getZ();
            // Assuming getVisibility() is the method to get the visibility of the landmark
            landmarkList[i++] = landmark.getVisibility();
        }
        return landmarkList;
    }


    private void initializeMediapipe() {
        AndroidAssetUtil.initializeNativeAssetManager(this);
        eglManager = new EglManager(null);
        // Create side packets map


        processor_pose_hand =
                new FrameProcessor(
                        this,
                        eglManager.getNativeContext(),
                        applicationInfo.metaData.getString("pose_hand_binary"),
                        applicationInfo.metaData.getString("inputVideoStreamName"),
                        applicationInfo.metaData.getString("outputVideoStreamName")

                );

        // Create an AndroidPacketCreator
        AndroidPacketCreator packetCreator = processor_pose_hand.getPacketCreator();
        Map<String, Packet> inputSidePackets = new HashMap<>();

        // Add the num_hands side packet
        inputSidePackets.put(INPUT_NUM_HANDS_SIDE_PACKET_NAME, packetCreator.createInt32(NUM_HANDS));

        inputSidePackets.put(INPUT_MODEL_COMPLEXITY,packetCreator.createInt32(applicationInfo.metaData.getInt("modelComplexity")));
        processor_pose_hand.setInputSidePackets(inputSidePackets);
        processor_pose_hand
                .getVideoSurfaceOutput()
                .setFlipY(applicationInfo.metaData.getBoolean("flipFramesVertically", FLIP_FRAMES_VERTICALLY));



        processor_pose_hand.addPacketCallback("pose_landmarks", (packet) -> {
            try {
                long timestamp = packet.getTimestamp();
                LandmarkData data = landmarkMap.getOrDefault(timestamp, new LandmarkData());
                assert data != null;

                if (PacketGetter.getProtoBytes(packet) != null) {
                    byte[] landmarksRaw = PacketGetter.getProtoBytes(packet);
                    NormalizedLandmarkList poseLandmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
                    data.poseLandmarks = convertLandmarksToListPose(poseLandmarks);
                    //Log.d(TAG, Arrays.toString(data.poseLandmarks));
                    onFrameProcessed();
                } else {
                    data.poseLandmarks = createZeroLandmarks(33,4); // 33 for pose landmarks
                }

                landmarkMap.put(timestamp, data);
                //checkAndConcatenateLandmarks(timestamp);
            } catch (Exception e) {
                Log.d(TAG, "Error from pose_landmarks callback: " + e);
            }

        });
        processor_pose_hand.addPacketCallback("hand_landmarks", (packet) -> {
            try {
                long timestamp = packet.getTimestamp();
                LandmarkData data = landmarkMap.getOrDefault(timestamp, new LandmarkData());
                assert data != null;


                    //byte[] landmarksRaw = PacketGetter.getProtoBytes(packet);
                    List<NormalizedLandmarkList> multiHandLandmarks = PacketGetter.getProtoVector(packet, NormalizedLandmarkList.parser());

                    // Check if there are landmarks for at least one hand
                    if (!multiHandLandmarks.isEmpty()) {
                        // If there's at least one hand detected, process the first hand (could be left or right)
                        data.leftHandLandmarks = convertLandmarksToList(multiHandLandmarks.get(0));
                        //Log.d(TAG, Arrays.toString(data.leftHandLandmarks));

                        // If there are two hands detected, process the second hand
                        if (multiHandLandmarks.size() > 1) {
                            data.rightHandLandmarks = convertLandmarksToList(multiHandLandmarks.get(1));
                        } else {
                            // If only one hand is detected, fill the second hand's data with zeros
                            data.rightHandLandmarks = createZeroLandmarks(21, 3);
                        }
                    } else {
                        // If no hands are detected, fill both hands' data with zeros
                        data.leftHandLandmarks = createZeroLandmarks(21, 3);
                        data.rightHandLandmarks = createZeroLandmarks(21, 3);
                    }



                landmarkMap.put(timestamp, data);
                checkAndConcatenateLandmarks(timestamp);


            } catch (Exception e) {
                Log.d(TAG, "Error from hand_landmarks callback: " + e);
            }
        });

    }
    // Convert the queue of float arrays to a single 2D float array
    private float[][] convertQueueToArray(Deque<float[]> queue) {
        float[][] array = new float[queue.size()][];
        return queue.toArray(array);
    }

    void checkAndConcatenateLandmarks(long timestamp) {
        LandmarkData data = landmarkMap.get(timestamp);
        if (data != null) {
            float[] concatenatedLandmarks = new float[258]; // 33 * 4 + 21 * 3 + 21 * 3
            System.arraycopy(data.poseLandmarks, 0, concatenatedLandmarks, 0, data.poseLandmarks.length);
            System.arraycopy(data.leftHandLandmarks, 0, concatenatedLandmarks, data.poseLandmarks.length, data.leftHandLandmarks.length);
            System.arraycopy(data.rightHandLandmarks, 0, concatenatedLandmarks, data.poseLandmarks.length + data.leftHandLandmarks.length, data.rightHandLandmarks.length);

            // Add to queue and remove old if necessary
                landmarksQueue.addLast(concatenatedLandmarks);

            if (landmarksQueue.size() > 95) {
                landmarksQueue.removeFirst();
            }

            // If there are enough data points, make a prediction
            if (landmarksQueue.size() == 95) {
                //Log.d(TAG, "attempting to make prediction "+ timestamp);
                //Log.d(TAG, String.valueOf(convertQueueToArray(landmarksQueue)[0].length));
                makePredictionAndUpdateUI(convertQueueToArray(landmarksQueue));
            }


            landmarkMap.remove(timestamp); // Remove the entry after processing
        }
    }
    private void initializeInterpreter() {
        try {
            // Initialize interpreter with CPU delegate

            Interpreter.Options options = new Interpreter.Options();
            FlexDelegate flexDelegate = new FlexDelegate();

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {

                options.addDelegate(flexDelegate);
                options.setNumThreads(8);
            }
            else{
                Log.e(TAG, "SDK not supported: ");
            }




            //Set the interpreter
            tflite = new Interpreter(FileUtil.loadMappedFile(this, "Sign.tflite"), options);


        } catch (Exception e) {
            Log.e(TAG, "Error initializing TFLite interpreter: " + e);
            tflite = null; // Ensure interpreter is null if initialization fails
        }
    }
    private void makePredictionAndUpdateUI(float[][] inputData) {

        // Reshape inputData to 4D: [1, 95, 258, 1]
        float[][][] reshapedInput = new float[1][inputData.length][inputData[0].length];

        for (int i = 0; i < inputData.length; i++) {
            for (int j = 0; j < inputData[0].length; j++) {
                reshapedInput[0][i][j] = inputData[i][j];
            }
        }
        //Test input dims
       //Log.d(TAG, "inputData dimensions: " + inputData.length + "x" + inputData[0].length);
       //Log.d(TAG, "reshapedInput dimensions: " + reshapedInput.length + "x" + reshapedInput[0].length + "x" + reshapedInput[0][0].length );


        // Assuming inputData is shaped correctly
        float[][] output = new float[1][10]; // As per model's output
        tflite.run(reshapedInput, output);

        // Process output to get gesture
        int maxIndex = getMaxIndex(output[0]);
        float maxConfidence = output[0][maxIndex];
        if (maxConfidence > 0.7) {
            String predictedGesture = gestures[maxIndex];
            updatePredictions(predictedGesture);
            updateTextView();
            //Log.d(TAG, "Successfully updated View");
        }
    }

    private void updatePredictions(String gesture) {
        if (lastPredictions.size() >= 5) {
            lastPredictions.removeFirst();
        }
        if (gesture!=lastPredictedGesture){
            lastPredictions.addLast(gesture);
        }

        lastPredictedGesture = gesture;
    }
    public void onFrameProcessed() {
        frameCount++;
        long currentTime = System.currentTimeMillis();
        long elapsedTime = currentTime - startTime;

        if (elapsedTime >= 1000) { // Every 1 second
            float fps = (float) frameCount / (elapsedTime / 1000.0f);
            Log.d(TAG, "FPS: " + fps);
            fps_meter.setText("fps= "+fps);
            startTime = System.currentTimeMillis();
            frameCount = 0;
        }
    }
    private void updateTextView() {
        StringBuilder text = new StringBuilder();
        for (String gesture : lastPredictions) {
            text.append(gesture).append("  ");
        }
        translatedText.setText(text.toString());
    }

    private int getMaxIndex(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

}





