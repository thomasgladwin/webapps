# set FLASK_APP=app
# set FLASK_ENV=development
# flask run

from flask import Flask, render_template, request
import semsim_funcs
import conttag_funcs

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
    targets_with_sims = []
    for z in zip(target_words, relative_sims):
        this_str = ' '.join([z[0], str(round(z[1], 3))])
        targets_with_sims.append(this_str)
    return render_template('semsim_results.html', targets_with_sims=targets_with_sims, target_words=target_words, pos_words=pos_words, neg_words=neg_words, relative_sims=relative_sims)

@app.route('/conttag')
def conttag():
    return render_template('conttag.html')

@app.route('/conttag_results', methods=['GET', 'POST'])
def conttag_results():
    text = request.form['text']
    tags = request.form['tags']
    tags_in_vocab, output_all, output_sim_all, output_per_paragraph, output_sim_per_paragraph, output_paragraphs = conttag_funcs.get_conttags(text, tags)
    tags_in_vocab_str = ','.join(tags_in_vocab)
    output_per_paragraph_str = ','.join(output_per_paragraph)
    output_sim_per_paragraph_str = ','.join([str(round(a, 3)) for a in output_sim_per_paragraph])
    text = text.split('\n')
    output_tag_with_sim_per_paragraph = []
    for z in zip(output_per_paragraph, output_sim_per_paragraph):
        this_str = ' '.join([z[0], str(round(z[1], 3))])
        output_tag_with_sim_per_paragraph.append(this_str)
    output_tag_with_sim_all = []
    for z in zip(output_all, output_sim_all):
        this_str = ' '.join([z[0], str(round(z[1], 3))])
        output_tag_with_sim_all.append(this_str)
    nested_info_per_paragraph = []
    for z in zip(output_per_paragraph, output_sim_per_paragraph, output_paragraphs):
        this_row = [z[0], round(z[1], 3), z[2]]
        nested_info_per_paragraph.append(this_row)
    return render_template('conttag_results.html', nested_info_per_paragraph=nested_info_per_paragraph,output_tag_with_sim_per_paragraph=output_tag_with_sim_per_paragraph, output_tag_with_sim_all=output_tag_with_sim_all,tags_in_vocab_str=tags_in_vocab_str, text=text, output_all=output_all, output_sim_all=output_sim_all, output_per_paragraph_str=output_per_paragraph_str, output_sim_per_paragraph_str=output_sim_per_paragraph_str)
