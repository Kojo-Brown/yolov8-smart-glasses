pluginManagement {

    repositories {

        google {

            content {

                includeGroupByRegex("com\\.android.*")

                includeGroupByRegex("com\\.google.*")

                includeGroupByRegex("androidx.*")

            }

        }

        mavenCentral()

        gradlePluginPortal()

        flatDir {

            dirs("libs") // For local AAR files if needed

        }

    }

}



dependencyResolutionManagement {

    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)

    repositories {

        google()

        mavenCentral()

        maven { url = uri("https://jitpack.io") }  // For Intel RealSense library

        maven { url = uri("https://maven.google.com") }  // For additional Android libraries

        }

        }



        rootProject.name = "YOLOv8 TfLite"

        include(":app")