from flask import Flask, render_template, jsonify, request
from core.transfer import style_transfer

SECRET_KEY = "you never know"
VGG_WEIGHT = "./models/vgg19_weights_normalized.h5"
DECODER_WEIGHT = "./models/decoder_weights.h5"
OUTPUT_DIR = './static/cache/stylized'
CONTENT_DIR = './static/cache/content'
STYLE_DIR = './static/cache/style'

app = Flask(__name__)
app.config.from_object(__name__)


@app.errorhandler
def not_found():
    return "404"


@app.route('/')
def main():
    return render_template("index.html")


@app.route('/transfer', methods=["POST"])
def transfer():
    input_content = request.files["input_content"]
    input_style = request.files["input_style"]
    name, path = style_transfer(content_img=input_content,
                                style_img=input_style,
                                content_dir=app.config['CONTENT_DIR'],
                                style_dir=app.config['STYLE_DIR'],
                                output_dir=app.config["OUTPUT_DIR"])
    return jsonify(name=name, path='.' + path)


if __name__ == "__main__":
    app.run(port=8080)
