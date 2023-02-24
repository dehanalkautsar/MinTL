import csv
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import math, re, argparse
import json
import functools
import pickle
from utils import clean_replace

order_to_number = {
    'first': 1, 'one': 1, 'seco': 2, 'two': 2, 'third': 3, 'three': 3, 'four': 4, 'forth': 4, 'five': 5, 'fifth': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nin': 9, 'ten': 10, 'eleven': 11, 'twelve': 12
}

def similar(a,b):
    return a == b or a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]
    #return a == b or b.endswith(a) or a.endswith(b)    

def setsub(a,b):
    junks_a = []
    useless_constraint = ['temperature','week','est ','quick','reminder','near']
    for i in a:
        flg = False
        for j in b:
            if similar(i,j):
                flg = True
        if not flg:
            junks_a.append(i)
    for junk in junks_a:
        flg = False
        for item in useless_constraint:
            if item in junk:
                flg = True
        if not flg:
            return False
    return True

def setsim(a,b):
    a,b = set(a),set(b)
    return setsub(a,b) and setsub(b,a)



class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):

        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


def report(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        args[0].metric_dict[func.__name__ + ' '+str(args[2])] = res
        return res
    return wrapper

class CamRestEvaluator(object):
	def __init__(self, reader):
		self.reader = reader
		self.entity_dict = self.reader.entity_dict
		self.all_data = self.reader.train + self.reader.dev + self.reader.test
		self.test_data = self.reader.test

		self.bleu_scorer = BLEUScorer()
		#self.nlp = spacy.load('en_core_web_sm')
		self.all_info_slot = [ 'area', 'food', 'pricerange']

		# only evaluate these slots for dialog success
		self.requestables = ['area','food','pricerange','postcode','phone','address']

	def pack_dial(self, data):
		dials = {}
		for turn in data:
			dial_id = turn['dial_id']
			if dial_id not in dials:
				dials[dial_id] = []
			dials[dial_id].append(turn)
		return dials
	
	def clean(self,s):
		s = s.replace('<go> ', '').replace(' SLOT', '_SLOT')
		# s = '<GO> ' + s + ' </s>'
		for item in self.entity_dict:
			s = clean_replace(s, item, '{}_SLOT'.format(self.entity_dict[item]))
		return s

	def validation_metric(self, data):
		for i,row in enumerate(data):
			data[i]['resp_gen'] = self.clean(data[i]['resp_gen'])
			data[i]['resp'] = self.clean(data[i]['resp'])
		
		bleu = self.bleu_metric(data)
		match = self.match_metric(data)
		success = self.success_f1_metric(data)
		
		return bleu, success, match[0]

	def bleu_metric(self,data,type='bleu'):
		gen, truth = [],[]
		for row in data:
			gen.append(row['resp_gen'])
			truth.append(row['resp'])
		wrap_generated = [[_] for _ in gen]
		wrap_truth = [[_] for _ in truth]
		sc = BLEUScorer().score(zip(wrap_generated, wrap_truth))
		return sc

	def match_metric(self, data, sub='match'):
		dials = self.pack_dial(data)
		match,total = 0,1e-8
		success = 0
		# find out the last placeholder and see whether that is correct
		# if no such placeholder, see the final turn, because it can be a yes/no question or scheduling dialogue
		for dial_id in dials:
			truth_req, gen_req = [], []
			dial = dials[dial_id]
			gen_bspan, truth_cons, gen_cons = None, None, set()
			truth_turn_num = -1
			truth_response_req = []
			for turn_num,turn in enumerate(dial):
				if 'SLOT' in turn['resp_gen']:
					gen_bspan = turn['bspn_gen']
					gen_cons = self._extract_constraint(gen_bspan)
				if 'SLOT' in turn['resp']:
					truth_cons = self._extract_constraint(turn['bspn'])
				gen_response_token = turn['resp_gen'].split()
				response_token = turn['resp'].split()
				for idx, w in enumerate(gen_response_token):
					if w.endswith('SLOT') and w != 'SLOT':
						gen_req.append(w.split('_')[0])
					if w == 'SLOT' and idx != 0:
						gen_req.append(gen_response_token[idx - 1])
				for idx, w in enumerate(response_token):
					if w.endswith('SLOT') and w != 'SLOT':
						truth_response_req.append(w.split('_')[0])
			if not gen_cons:
				gen_bspan = dial[-1]['bspn_gen']
				gen_cons = self._extract_constraint(gen_bspan)
			if truth_cons:
				if gen_cons == truth_cons:
					match += 1
				
				total += 1

		#try print(total)
		return match / total, success / total
	
	def success_f1_metric(self, data, sub='successf1'):
		dials = self.pack_dial(data)
		tp,fp,fn = 0,0,0
		for dial_id in dials:
			truth_req, gen_req = set(),set()
			dial = dials[dial_id]
			for turn_num, turn in enumerate(dial):
				gen_response_token = turn['resp_gen'].split()
				response_token = turn['resp'].split()
				for idx, w in enumerate(gen_response_token):
					if w.endswith('SLOT') and w != 'SLOT':
						gen_req.add(w.split('_')[0])
				for idx, w in enumerate(response_token):
					if w.endswith('SLOT') and w != 'SLOT':
						truth_req.add(w.split('_')[0])

			gen_req.discard('name')
			truth_req.discard('name')
			for req in gen_req:
				if req in truth_req:
					tp += 1
				else:
					fp += 1
			for req in truth_req:
				if req not in gen_req:
					fn += 1
		precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
		f1 = 2 * precision * recall / (precision + recall + 1e-8)
		return f1

	def _extract_constraint(self, z):
		z = z.split()
		if 'EOS_Z1' not in z:
			s = set(z)
		else:
			idx = z.index('EOS_Z1')
			s = set(z[:idx])
		if 'moderately' in s:
			s.discard('moderately')
			s.add('moderate')
		return s.intersection(self.reader.entities)	