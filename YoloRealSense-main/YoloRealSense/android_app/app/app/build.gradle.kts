plugins {

    id("com.android.application")

    id("org.jetbrains.kotlin.android")

}



android {

    namespace = "com.surendramaran.yolov8tflite"

    compileSdk = 35



    defaultConfig {

        applicationId = "com.surendramaran.yolov8tflite"

        minSdk = 26  // Increased for RealSense compatibility

        targetSdk = 30

        versionCode = 1

        versionName = "1.0"



        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {

            abiFilters += listOf("armeabi-v7a", "arm64-v8a")

        }

    }



    buildTypes {

        release {

            isMinifyEnabled = false

            proguardFiles(

                getDefaultProguardFile("proguard-android-optimize.txt"),

                "proguard-rules.pro"

            )

        }

    }



    compileOptions {

        sourceCompatibility = JavaVersion.VERSION_1_8

        targetCompatibility = JavaVersion.VERSION_1_8

    }



    kotlinOptions {

        jvmTarget = "1.8"

    }



    buildFeatures {

        viewBinding = true

    }



    packagingOptions {

        resources.excludes += setOf(

            "META-INF/DEPENDENCIES",

            "META-INF/LICENSE",

            "META-INF/*.kotlin_module"

        )

        jniLibs.useLegacyPackaging = true

        pickFirsts += setOf(

            "**/librealsense.so",

            "**/libc++_shared.so"

        )

    }

}



dependencies {

    // Core Android

    implementation("androidx.core:core-ktx:1.16.0")

    implementation("androidx.appcompat:appcompat:1.7.1")

    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.1")

    implementation("androidx.activity:activity-ktx:1.10.1")



    // UI

    implementation("androidx.constraintlayout:constraintlayout:2.2.1")

    implementation("com.google.android.material:material:1.12.0")



    // RealSense (using JitPack)

    implementation(files("libs/librealsense-release.aar"))




    // Audio Features

    implementation("androidx.media:media:1.7.0")



    // CameraX (latest stable version)

    val cameraxVersion = "1.3.1"

    implementation("androidx.camera:camera-core:$cameraxVersion")

    implementation("androidx.camera:camera-camera2:$cameraxVersion")

    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")

    implementation("androidx.camera:camera-view:$cameraxVersion")



    // TensorFlow Lite

    implementation("org.tensorflow:tensorflow-lite:2.14.0")

    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")

    implementation("org.tensorflow:tensorflow-lite-task-vision:0.4.4")

    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")


    // ML Kit Text Recognition
    implementation("com.google.mlkit:text-recognition:16.0.1")

    // Testing

    testImplementation("junit:junit:4.13.2")

    androidTestImplementation("androidx.test.ext:junit:1.2.1")

    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")

}


//implementation(files("libs/librealsense-release.aar"))