pip_install () {
    virtualenv venv
    source venv/bin/activate

    pip install -r requirements.txt

    deactivate
}

start_jupyter() {
    source venv/bin/activate

    jupyter notebook ./notebooks/pb_interp_workshop.ipynb
}

main () {
    pip_install

    start_jupyter
}

main
