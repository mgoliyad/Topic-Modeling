from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.phrases import Phrases
from gensim.models.word2vec import LineSentence
from gensim.utils import simple_preprocess
from gensim.corpora.mmcorpus import MmCorpus
from pprint import pprint
import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from pathlib import Path
import json
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#stop_words.extend(['xxx'])
import pyLDAvis
import pyLDAvis.gensim_models


class Topic_modeling():
 
    def __init__(self, output_dir, config, printout):
        self.min_text_length = config["topic_modeling"]["min_text_length"]
        self.chunksize = config["topic_modeling"]["chunksize"]
        self.passes = config["topic_modeling"]["passes"]
        self.iterations = config["topic_modeling"]["iterations"]
        self.chunksize = config["topic_modeling"]["chunksize"]
        self.min_k = config["topic_modeling"]["min_k"]
        self.max_k = config["topic_modeling"]["max_k"] + 1
        self.printout = printout
        self.output_dir = output_dir


    def format_topics_sentences(self, ldamodel, corpus, all_texts):
        sent_topics_df = pd.DataFrame(columns = ['class_name', 'score'])
    
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            (topic_num, topic_score) = row[0]
            top_words = ldamodel.show_topic(topic_num)
            topic_name = top_words[0][0] + '_' + top_words[1][0] + '_' + str(topic_num)
            sent_topics_df.loc[len(sent_topics_df)] = [topic_name, round(topic_score,4)]
    
        sent_topics_df['text'] = [' '.join(ws)[:250] for ws in all_texts.values()]
        sent_topics_df['filename'] = all_texts.keys()
        return sent_topics_df
    
     
    
    def gensim_LDA(self, data_dir):
        '''Generate dictionary using trigrams; if Dictionary file is foind then upload dictionary from there.'''

        start_dt = dt.datetime.now()
        gensim_dir =  Path(data_dir.parent, 'GENSIM')
        gensim_dir.mkdir(parents=True, exist_ok=True)
        dictionary_filepath = Path(gensim_dir, 'trigram_dict.dict')
        bow_corpus_filepath = Path(gensim_dir, 'bow_trigrams_corpus.mm')
        trigrams_filepath = Path(gensim_dir,  'trigrams_all.txt')
        all_texts_filepath = Path(gensim_dir,  'all_texts.json')
    
    
        if trigrams_filepath.exists():
    
            f = open(trigrams_filepath, 'r')
            trigram_all_words = json.load(f)
            f = open(all_texts_filepath, 'r')
            all_texts = json.load(f)
            all_words = all_texts.values()
        else:
            all_texts ={}
            for s_dir in data_dir.iterdir():
                if s_dir.is_dir():
                    for f in s_dir.iterdir():
                        for sent in LineSentence(f):
                            words = [word for word in simple_preprocess(str(sent), deacc=True) if word not in stop_words]
                            if len(words) < self.min_text_length:
                                continue
                        all_texts[str(f)] = words
    
            with open(all_texts_filepath, 'w') as f:
                json.dump(all_texts, f)
    
            all_words = all_texts.values()
            bigram = Phrases(all_words)
            trigram = Phrases(bigram[all_words], threshold=1)  
            trigram_all_words = [trigram[bigram[w]] for w in all_words]
            with open(trigrams_filepath, 'w') as f:
                json.dump(trigram_all_words, f)
    
        if dictionary_filepath.exists():
            dictionary = Dictionary.load(str(dictionary_filepath))
            corpus = MmCorpus(str(bow_corpus_filepath))
            
        else:
            dictionary = Dictionary(trigram_all_words)
            dictionary.filter_extremes(no_below=20, no_above=0.5)
            corpus = [dictionary.doc2bow(doc) for doc in all_texts.values()]
    
            def bow_generator(all_texts):
                for doc in all_texts.values():
                    yield dictionary.doc2bow(doc)
    
            MmCorpus.serialize(str(bow_corpus_filepath), bow_generator(all_texts))
            dictionary.save(str(dictionary_filepath))
    
        workers  = int(os.environ['NUMBER_OF_PROCESSORS']) -1
        avg_topic_coherences = {}
        frq = {}
        dist = {}
        best_score = -10
        best_frq = 100
        end_dt = dt.datetime.now()
        
        if self.printout == True:
            print('Corpus size {}, {} '.format(len(corpus), len(all_texts.values())))
            print('DONE with data preparation in {}'.format(end_dt - start_dt))
    
        '''Start training model for number of topics between min_k and max_k from config file.''' 
        for num_topics in range(self.min_k, self.max_k):
    
            if self.printout == True:
                print('num_topics = %d'%num_topics)
            model = LdaMulticore(
                corpus=corpus,
                id2word=dictionary,
                chunksize=self.chunksize,
                eta='auto',
                iterations=self.iterations,
                num_topics=num_topics,
                passes=self.passes,
                eval_every=None,
                workers=workers
            )
            
            '''Saving coherence score to compare across all topib_num'''
            
            c_v_top_topics = model.top_topics(corpus, texts=all_words, coherence='c_v', \
                                              processes=workers)
            
            avg_topic_coherence = sum([t[1] for t in c_v_top_topics]) / num_topics
            avg_topic_coherences[num_topics] = avg_topic_coherence

            if self.printout == True:
                print('Average topic coherence: %.4f.' % avg_topic_coherence)
    
           
            LDAvis_prepared = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
            topic_data = LDAvis_prepared[0]

            dist_xy = np.sort(distance_matrix(topic_data[['x', 'y']], topic_data[['x', 'y']]).flatten())
            dist_frq = np.sort(distance_matrix(topic_data[['Freq']], topic_data[['Freq']]).flatten())
            
            dist_xy = dist_xy[(dist_xy > 0)]
            dist_frq = dist_frq[(dist_frq > 0)]
            
            frq[num_topics] = np.max(dist_frq) - np.min(dist_frq)
            dist[num_topics] = np.min(dist_xy)
                
            if self.printout == True:
                print('Min distance between topics: {}'.format(dist[num_topics]))
                print('Diff between min and max diff of Freq of topics: {}'.format(frq[num_topics]))

            '''Generate html for current model'''
            html_path = Path(self.output_dir, 'LDA_with_' + str(num_topics) + '_topics.html')
            pyLDAvis.save_html(LDAvis_prepared, str(html_path))
            
           
            if avg_topic_coherence > best_score and frq[num_topics] < best_frq:
                best_frq = frq[num_topics]
                best_score = avg_topic_coherence
                best_model = model
                best_num_topics = num_topics
                if self.printout == True:
                    print('Best: %.4f, %d'%(best_score, num_topics))
    
        if (self.printout == True) and  (len(avg_topic_coherences) > 2):
            
            fig, (ax1, ax2, ax3) = plt.subplots(3)
            ax1.plot(avg_topic_coherences.keys(), avg_topic_coherences.values())
            ax1.set_title("Coherence score")
            ax2.plot(frq.keys(), frq.values())
            ax2.set_title("Diff between max and min Frq")
            ax3.plot(dist.keys(), dist.values())
            ax3.set_title("Minimum distance")
            fig.show()
    
        '''Generate output to show best text for each topic'''
        
        classified_texts = self.format_topics_sentences(ldamodel=best_model, corpus=corpus, all_texts=all_texts)
        classified_texts_sorted = pd.DataFrame()
        sent_topics_outdf_grpd = classified_texts.groupby('class_name')
    
        for i, grp in sent_topics_outdf_grpd:
            classified_texts_sorted = pd.concat([classified_texts_sorted,
                                                     grp.sort_values(['score'], ascending=[0]).head(1)],
                                                    axis=0) 
        #classified_texts_sorted.reset_index(drop=True, inplace=True)
    
        if self.printout == True:
            print(classified_texts_sorted.columns)
    
        for d in classified_texts_sorted.itertuples():
            pprint(d.class_name)
            pprint(d.score)
            pprint(d.text)
            if d.index == best_num_topics*3:
                break

        end_dt1 = dt.datetime.now()
        print('DONE with Topic modeling in %s' % (end_dt1 - end_dt))