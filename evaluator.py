import csv
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import math, re, argparse, logging
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
    useless_constraint = ['temperature','week','est ','quick','reminder','near','suhu','minggu','cepat','pengingat','dekat','terdekat']
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

	def run_metrics(self, data, eval_act=True):
		for i,row in enumerate(data):
			data[i]['resp_gen'] = self.clean(data[i]['resp_gen'])
			data[i]['resp'] = self.clean(data[i]['resp'])
		
		bleu = self.bleu_metric(data)
		match = self.match_metric(data)
		success = self.success_f1_metric(data)
		logging.info('[TEST PHASE] match: %2.5f  success: %2.5f  bleu: %2.5f'%(match[0], success, bleu))

		return [{'bleu':bleu, 'success':success, 'match':match[0]}]

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


class SMDEvaluator(object):
	def __init__(self, reader):
		self.reader = reader
		self.entity_dict = self.reader.entity_dict
		self.all_data = self.reader.train + self.reader.dev + self.reader.test
		self.test_data = self.reader.test

		self.bleu_scorer = BLEUScorer()
		#self.nlp = spacy.load('en_core_web_sm')
		self.all_info_slot = ['date','location','weather_attribute', 'poi_type', 'distance', 'event', 'time', 'agenda', 'party', 'room']

		# only evaluate these slots for dialog success
		self.requestables = ['weather_attribute', 'poi', 'traffic_info', 'address', 'distance', 'date', 'time', 'party', 'agenda', 'room']

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

	def run_metrics(self, data, eval_act=True):
		for i,row in enumerate(data):
			data[i]['resp_gen'] = self.clean(data[i]['resp_gen'])
			data[i]['resp'] = self.clean(data[i]['resp'])
		
		bleu = self.bleu_metric(data)
		match = self.match_metric(data)
		success = self.success_f1_metric(data)
		logging.info('[TEST PHASE] match: %2.5f  success: %2.5f  bleu: %2.5f'%(match, success, bleu))

		return [{'bleu':bleu, 'success':success, 'match':match}]

	def validation_metric(self, data):
		for i,row in enumerate(data):
			data[i]['resp_gen'] = self.clean(data[i]['resp_gen'])
			data[i]['resp'] = self.clean(data[i]['resp'])
		
		bleu = self.bleu_metric(data)
		match = self.match_metric(data)
		success = self.success_f1_metric(data)
		
		return bleu, success, match

	def bleu_metric(self,data,type='bleu'):
		gen, truth = [],[]
		for row in data:
			gen.append(row['resp_gen'])
			truth.append(row['resp'])
		wrap_generated = [[_] for _ in gen]
		wrap_truth = [[_] for _ in truth]
		sc = BLEUScorer().score(zip(wrap_generated, wrap_truth))
		return sc

	def _extract_constraint(self, z):
		z = z.split()
		if 'EOS_Z1' not in z:
			s = set(z)
		else:
			idx = z.index('EOS_Z1')
			s = set(z[:idx])
		reqs = ['address', 'traffic', 'poi', 'poi_type', 'distance', 'weather', 'temperature', 'weather_attribute',
				'date', 'time', 'location', 'event', 'agenda', 'party', 'room', 'weekly_time', 'forecast']
		informable = {
			'weather': ['date','location','weather_attribute'],
			'navigate': ['poi_type','distance'],
			'schedule': ['event', 'date', 'time', 'agenda', 'party', 'room']
		}
		infs = []
		for v in informable.values():
			infs.extend(v)
		junk = ['good','great','quickest','shortest','route','week','fastest','nearest','next','closest','way','mile',
				'activity','restaurant','appointment' ]
		s = s.difference(junk).difference(reqs)
		res = set()
		for item in s:
			if item in junk:
				continue
			flg = False
			for canon_ent in sorted(list(self.entity_dict.keys())):
				if self.entity_dict[canon_ent] in infs:
					if similar(item, canon_ent):
						flg = True
						junk.extend(canon_ent.split())
						res.add(canon_ent)
					if flg:
						break
		return res

	def constraint_same(self, truth_cons, gen_cons):
		if not truth_cons and not gen_cons:
			return True
		if not truth_cons or not gen_cons:
			return False
		return setsim(gen_cons, truth_cons)

	def _get_entity_dict(self, entity_data):
		entity_dict = {}
		for k in entity_data:
			if type(entity_data[k][0]) is str:
				for entity in entity_data[k]:
					# entity = self._lemmatize(self._tokenize(entity))
					entity_dict[entity] = k
					if k in ['event','poi_type']:
						entity_dict[entity.split()[0]] = k
			elif type(entity_data[k][0]) is dict:
				for entity_entry in entity_data[k]:
					for entity_type, entity in entity_entry.items():
						entity_type = 'poi_type' if entity_type == 'type' else entity_type
						# entity = self._lemmatize(self._tokenize(entity))
						entity_dict[entity] = entity_type
						if entity_type in ['event', 'poi_type']:
							entity_dict[entity.split()[0]] = entity_type
		self.entity_dict = entity_dict

	def match_metric(self, data, sub='match',bspans='./data/kvret/test.bspan.pkl'):
		dials = self.pack_dial(data)
		match,total = 0,1e-8
		#bspan_data = pickle.load(open(bspans,'rb'))
		# find out the last placeholder and see whether that is correct
		# if no such placeholder, see the final turn, because it can be a yes/no question or scheduling conversation
		for dial_id in dials:
			dial = dials[dial_id]
			gen_bspan, truth_cons, gen_cons = None, None, set()
			truth_turn_num = -1
			for turn_num,turn in enumerate(dial):
				if 'SLOT' in turn['resp_gen']:
					gen_bspan = turn['bspn_gen']
					gen_cons = self._extract_constraint(gen_bspan)
				if 'SLOT' in turn['resp']:
					truth_cons = self._extract_constraint(turn['bspn'])

			# KVRET dataset includes "scheduling" (so often no SLOT decoded in ground truth)
			if not truth_cons:
				truth_bspan = dial[-1]['bspn']
				truth_cons = self._extract_constraint(truth_bspan)
			if not gen_cons:
				gen_bspan = dial[-1]['bspn_gen']
				gen_cons = self._extract_constraint(gen_bspan)

			if truth_cons:
				if self.constraint_same(gen_cons, truth_cons):
					match += 1
					#print(gen_cons, truth_cons, '+')
				#else:
				#    #print(gen_cons, truth_cons, '-')
				total += 1

		return match / total

	# def _tokenize(self, sent):
	# 	return ' '.join(word_tokenize(sent))

	# def _lemmatize(self, sent):
	# 	words = [wn.lemmatize(_) for _ in sent.split()]
	# 	#for idx,w in enumerate(words):
	# 	#    if w !=
	# 	return ' '.join(words)

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