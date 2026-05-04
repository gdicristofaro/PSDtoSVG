from psd_tools import PSDImage
from flask import Flask, redirect, send_from_directory, abort, request, Response
from psdtosvg.psdtosvg import psd_to_svg


app = Flask(__name__, static_url_path='/PSDtoSVG', static_folder='static')

@app.route('/')
def root_redirect():
    '''Redirect the root URL to the /PSDtoSVG index page.'''
    return redirect('/PSDtoSVG')

@app.route('/PSDtoSVG')
def serve_psdtosvg_index():
    '''Serve the index.html file for the /PSDtoSVG route.'''
    return send_from_directory('static', 'index.html')

@app.route("/api/v1/upload", methods=['POST'])
def upload_file():
    '''Handle file upload and convert PSD to SVG.'''
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