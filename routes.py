from flask import Blueprint, render_template

routes = Blueprint("routes", __name__)

@routes.route("/")
def eeg_dashboard():
    return render_template("eeg.html")
