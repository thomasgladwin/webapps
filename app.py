# set FLASK_APP=app
# set FLASK_ENV=development
# flask run

from flask import Flask, render_template, request
import semsim_funcs
import semtag_funcs
import semnull_funcs
import textCoder_funcs

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
    contrast_word = request.form['contrast_word']
    pos_words = ''.join(request.form['pos_words'].lower().split()).split(',')
    neg_words = ''.join(request.form['neg_words'].lower().split()).split(',')
    relative_sims, target_words, pos_words, neg_words, error_message = semsim_funcs.get_sims(target_words, pos_words, neg_words, contrast_word=contrast_word)
    print(relative_sims)
    targets_with_sims = []
    for z in zip(target_words, relative_sims):
        this_str = ' '.join([z[0], str(round(z[1], 3))])
        targets_with_sims.append(this_str)
    return render_template('semsim_results.html', error_message=error_message, targets_with_sims=targets_with_sims, target_words=target_words, pos_words=pos_words, neg_words=neg_words, relative_sims=relative_sims, contrast_word=contrast_word)

@app.route('/semtag')
def semtag():
    return render_template('semtag.html')

@app.route('/semtag_results', methods=['GET', 'POST'])
def semtag_results():
    text = request.form['text']
    tags = request.form['tags']
    tags_in_vocab, output_all, output_sim_all, output_per_paragraph, output_sim_per_paragraph, output_paragraphs = semtag_funcs.get_semtags(text, tags)
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
    return render_template('semtag_results.html', nested_info_per_paragraph=nested_info_per_paragraph,output_tag_with_sim_per_paragraph=output_tag_with_sim_per_paragraph, output_tag_with_sim_all=output_tag_with_sim_all,tags_in_vocab_str=tags_in_vocab_str, text=text, output_all=output_all, output_sim_all=output_sim_all, output_per_paragraph_str=output_per_paragraph_str, output_sim_per_paragraph_str=output_sim_per_paragraph_str)

@app.route('/semnull')
def semnull():
    return render_template('semnull.html')

@app.route('/semnull_results', methods=['GET', 'POST'])
def semnull_results():
    target_words = ''.join(request.form['target_words'].lower().split()).split(',')
    contrast_word = request.form['contrast_word']
    pos_words = ''.join(request.form['pos_words'].lower().split()).split(',')
    neg_words = ''.join(request.form['neg_words'].lower().split()).split(',')
    scores_to_test = request.form['scores_to_test']
    template_sentence = request.form['template_sentence']
    template_pos = request.form['template_pos']
    print(target_words)
    print(contrast_word)
    null_distr_scores = semnull_funcs.get_semnull(pos_words, neg_words, template_sentence, template_pos)
    N_random_words_found = len(null_distr_scores)
    print(N_random_words_found)
    p_values, target_words_scores, target_words_output, scores_to_test_output, error_message = semnull_funcs.get_p(target_words, pos_words, neg_words, scores_to_test, null_distr_scores, contrast_word=contrast_word)
    p_values_nested = []
    target_words_output.extend(scores_to_test_output)
    print(p_values)
    for z in zip(target_words_output, target_words_scores, p_values):
        p_values_nested.append([z[0], round(z[1], 3), z[2]])
    print(p_values_nested)
    return render_template('semnull_results.html', error_message=error_message,null_distr_scores=null_distr_scores, N_random_words_found=N_random_words_found, p_values_nested=p_values_nested)

@app.route('/textCoder')
def textCoder():
    return render_template('textCoder.html')

@app.route('/textCoder_results', methods=['GET', 'POST'])
def textCoder_results():
    text = request.form['text']
    print(text)
    knowledge_list = textCoder_funcs.get_textCoder(text)
    topics_str = ','.join([k[0] for k in knowledge_list])
    attributes_str = ', '.join([a[0] for k in knowledge_list for a in k[1]])
    text_in_paragraphs = text.split("\n")
    nested_info_per_paragraph = []
    nested_info_per_paragraph_count = []
    nested_info_per_paragraph_parindex = []
    topics_str_list = [k[0] for k in knowledge_list]
    attr_str_list = [', '.join([a[0] for a in k[1]]) for k in knowledge_list]
    for z in zip(topics_str_list, attr_str_list):
        nested_info_per_paragraph.append([z[0], z[1]])
    attr_str_count_list = [', '.join([a[0] + " (" + str(a[1]) + ")" for a in k[1]]) for k in knowledge_list]
    for z in zip(topics_str_list, attr_str_count_list):
        nested_info_per_paragraph_count.append([z[0], z[1]])
    attr_str_parindex_list = [', '.join([a[0] + " (" + str(a[2]) + ")" for a in k[1]]) for k in knowledge_list]
    for z in zip(topics_str_list, attr_str_parindex_list):
        nested_info_per_paragraph_parindex.append([z[0], z[1]])
    return render_template('textCoder_results.html', topics_str=topics_str, attributes_str=attributes_str, nested_info_per_paragraph=nested_info_per_paragraph, text_in_paragraphs=text_in_paragraphs,nested_info_per_paragraph_count=nested_info_per_paragraph_count,nested_info_per_paragraph_parindex=nested_info_per_paragraph_parindex)
