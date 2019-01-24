pip_install () {
    pip install virtualenv==15.2.0
    virtualenv -p python3.6 venv
    source venv/bin/activate

    pip install -r requirements.txt

    deactivate
}

start_jupyter() {
    source venv/bin/activate

    jupyter notebook ./notebooks/student_thursday_workshop_full.ipynb
}

main () {
    pip_install

    start_jupyter
}

main
