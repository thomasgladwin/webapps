# set FLASK_APP=app
# set FLASK_ENV=development
# flask run

from flask import Flask, render_template, request
import semsim_funcs
import emocont_funcs

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/semsim')
def semsim():
    return render_template('semsim.html')

@app.route('/semsim_results', methods=['GET', 'POST'])
def semsim_results():
    target_words = ''.join(request.form['target_words'].lower().split()).split(',')
    pos_words = ''.join(request.form['pos_words'].lower().split()).split(',')
    neg_words = ''.join(request.form['neg_words'].lower().split()).split(',')
    relative_sims, target_words, pos_words, neg_words = semsim_funcs.get_sims(target_words, pos_words, neg_words)
    print(relative_sims)
    return render_template('semsim_results.html', target_words=target_words, pos_words=pos_words, neg_words=neg_words, relative_sims=relative_sims)

@app.route('/emocont')
def emocont():
    return render_template('emocont.html')

@app.route('/emocont_results', methods=['GET', 'POST'])
def emocont_results():
    text = request.form['text']
    output_all, output_per_paragraph = emocont_funcs.get_emocont(text)
    output_per_paragraph_str = ','.join(output_per_paragraph)
    return render_template('emocont_results.html', text=text, output_all=output_all, output_per_paragraph_str=output_per_paragraph_str)
