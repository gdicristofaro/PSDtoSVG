from psd_tools import PSDImage
from flask import Flask, render_template, abort, request, Response
from psdtosvg.psdtosvg import psd_to_svg


app = Flask(__name__)
 
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/svgmanipulator.html")
def manipulator():
    return render_template('svgmanipulator.html')

@app.route("/animations.html")
def animations():
    return render_template('animations.html')

@app.route("/upload", methods=['POST'])
def upload_file():
    try:
        psd_file = request.files['psd_file'].stream
        psd = PSDImage.open(psd_file)
    except Exception as e:
        print("Unable to parse uploaded file as PSD." + str(e))
        abort(500, "Unable to parse uploaded file as PSD.")

    try:
        svg_str = psd_to_svg(psd).encode('utf8')
        return Response(svg_str, mimetype='image/svg+xml')
    except Exception as e:
        print("Unable to handle request." + str(e))
        abort(500, "Unable to handle request.")


if __name__ == "__main__":
    app.run(debug=True)