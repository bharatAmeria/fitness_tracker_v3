pipeline {
    agent any  // Run on any available agent

    environment {
        VENV_DIR = "tracker"  // Define virtual environment path
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/bharatAmeria/fitness_tracker_v3.git'
            }
        }

        stage('Setup Environment') {
            steps {
                sh 'python3 -m venv $VENV_DIR'
                sh './tracker/bin/pip install -r requirements.txt'
            }
        }

        stage('Train Model') {
            steps {
                sh './tracker/bin/python app.py'
            }
        }
    }  // ✅ Properly closed 'stages' block

    post {
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check logs.'
        }
    }
}
