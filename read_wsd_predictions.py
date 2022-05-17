
import nltk
from nltk.corpus import wordnet as wn
from collections import Counter
import numpy as np
import argparse
import pdb
from scipy.stats import entropy as calculate_entropy
from time import time
import random
from utils import *


random.seed(9)

def find_lemmapos_in_xmldict(root, instance_id):
	text_id, sentence_id, _ = instance_id.split(".")
	text = root[text_id]
	for sentence in text:
		if sentence.attrib['id'] == ".".join([text_id, sentence_id]):
			for token in sentence:
				if 'id' in token.attrib and token.attrib['id'] == instance_id:
					lemma, pos = token.attrib['lemma'], token.attrib['pos']
					tk = token.text
					return tk, lemma, pos

def partially_transform_xml_to_dict(root):
	texts = dict()
	for text in root:
		texts[text.attrib["id"]] = text
	return texts




def join_info_and_predictions(dataset_name, predictions, measure):
	# Read the xml: loop over the xml file and keep track of instance information.
	# if the measure is avgsenses, pa or entropy we want to include monosemous words in the output. (which are not marked as "instances" but as "wf", and they dont have a prediction)
	def meets_condition(token, pos, measure, instance_dict):
		if 'id' in token.attrib and token.attrib['id'] in instance_dict: # If the instance is annotated with a sense, keep it
			#I 			am			for sure interested in this instance, regardless of the measure that I am using
			# ATTENTION! This varies for semeval2015 and semeval-gold... when using bary I don't care if there is no prediction
			return True
		# if there is no annotation available:
		# keep the instance anyway if we are calculating PA (Potential ambiguity) since this measure does not need annotations
		elif "id" in token.attrib and measure == "pa" and token.attrib['id'] not in instance_dict:
			return True
		elif 'id' not in token.attrib:
			if measure in ["avgsenses", "pa", "entropy"] and pos in ['NOUN', 'VERB', 'ADJ', 'ADV']: # keep this too, they can be monosemous and these measures take mono words into account
				return True # (and we will omit adverbs later)
		return False # if it does not meet any conditions above, we do not want to keep it

	detailed_instances = []
	print("load dataset:")
	df_data = load_and_prepare_dataset(dataset_name)
	# parse xml
	root = open_xml(dataset_name)
	root = partially_transform_xml_to_dict(root)
	# build the complete dictionary...
	print("total # predictions:", len(predictions))
	instanceid_predictions_dict = dict()
	for instanceid_prediction in predictions:
		if len(instanceid_prediction) < 2:
			continue
		else:
			idi = instanceid_prediction[0]
			prediction = "#".join(sorted(instanceid_prediction[1:]))
			instanceid_predictions_dict[idi] = prediction

	print("# of instances without prediction:", len(predictions)-len(instanceid_predictions_dict))

	t0 = time()
	texts_doing = 0  # texts correspond to utterances
	for text_id in root:
		texts_doing +=1
		if texts_doing % 1000 == 0:
			t1 = time()
			print("did", texts_doing, "texts in", t1 - t0)
			print(texts_doing)
		text = root[text_id]
		for sentence in text:
			sentence_id = sentence.attrib['id']
			actual_text_id = determine_text_id_from_sentence_id(sentence_id, dataset_name)
			utterance = df_data[df_data["RowID"] == actual_text_id].iloc[0]
			for token in sentence:
				lemma, pos = token.attrib['lemma'], token.attrib['pos']
				if meets_condition(token, pos, measure, instanceid_predictions_dict):
					tk = token.text
					new_item = dict({"token": tk, "lemma": lemma, "pos": pos})
					if 'id' in token.attrib:
						instance_id = token.attrib['id']
						if instance_id in instanceid_predictions_dict:
							new_item["instance_id"] = instance_id
							new_item["sense"] = instanceid_predictions_dict[instance_id]

					for col in df_data.columns:
						if col not in new_item:
							new_item[col] = utterance[col]
					detailed_instances.append(new_item)
	return detailed_instances, df_data


def find_number_of_senses(lemma, pos, sense_type):
	synsets = wn.synsets(lemma, posmap_u_to_wn(pos))
	if sense_type == "senses":
		senses = synsets
	elif sense_type == "supersenses":
		senses = set([ss.lexname() for ss in synsets])
	elif sense_type == "hypernyms":
		senses = set([tuple(sorted(ss.hypernyms())) for ss in synsets if ss.hypernyms()])
	num_of_senses = len(senses)
	return num_of_senses, synsets


def find_polysemous_wordpos(data, sense_type, measure):
	'''Find words that are polysemous (>1 sense) according to the given sense_type. Depending on the measure, we include also monosemous words.
	 Return a list of dicts. Each dict is an instance
	of a polysemous word. Here we do not exclude adverbs yet.
	if the measure is potential ambiguity (pa), we include monosemous words'''
	poly_words_data = []
	if measure == "pa": # potential ambiguity: we keep monosemous words AND disregard sense annotation
		searched = dict()
		for instance in data:
			if (instance['lemma'],instance['pos']) in searched:
				num_of_senses = searched[instance['lemma'],instance['pos']]
			else:
				num_of_senses = find_number_of_senses(instance['lemma'],instance['pos'], sense_type)[0]
				searched[(instance['lemma'], instance['pos'])] = num_of_senses
			if num_of_senses:
				poly_words_data.append(instance)
		return poly_words_data

	poly_lemmapos = set()
	mono_lemmapos = set()
	for winstance in data:
		lemma, pos = winstance['lemma'], winstance['pos']

		if (lemma, pos) in poly_lemmapos and "sense" in winstance:  # if we already know that it is polysemous (and there is a sense annotation available)
			poly_words_data.append(winstance)
		elif (lemma, pos) not in poly_lemmapos:  # if it's not a known poly word:
			if (lemma, pos) in mono_lemmapos and measure != "avgsenses" and measure != "entropy":  # if we know it's monosemous and measure is mosd, ignore it
				continue
			num_of_senses, senses = find_number_of_senses(lemma, pos, sense_type) # calculate its number of senses

			if num_of_senses > 1 and "sense" in winstance:  # if it is polysemous and a sense annotation is available:
				poly_lemmapos.add((lemma, pos))  # store the info
				poly_words_data.append(winstance)
			elif num_of_senses == 1: # if it is monosemous (we assume it is used in its only possible sense)
				mono_lemmapos.add((lemma, pos)) # store the info
				if "avgsenses" in measure or "entropy" in measure :  # these two measures take into account monosemous words and need sense annotations
					the_lemma = [l for l in senses[0].lemmas() if l._name == lemma]
					if the_lemma:
						onlysenseid = the_lemma[0].key()
					else:
						onlysenseid = senses[0].lemmas()[0].key()

					winstance['sense'] = onlysenseid
					poly_words_data.append(winstance)

	return poly_words_data


def find_repeated_words(data, include_adverbs, min_frequency=2):
	''' Select instances of lemma-pos that are used at least min_frequency times in the corpus'''
	lemmaposs = [(x["lemma"], x["pos"]) for x in data]
	c = Counter(lemmaposs)

	if include_adverbs:
		repeated_lemmapos = [x for x in c if c[x] >= min_frequency]
	else:
		repeated_lemmapos = [x for x in c if c[x] >= min_frequency and x[1] != "ADV"]

	kept_repeated_lemmapos = repeated_lemmapos

	instances_of_repeated_lemmaposs = [x for x in data if (x["lemma"], x["pos"]) in kept_repeated_lemmapos]

	return instances_of_repeated_lemmaposs, c


def mosd_measure(data, counter, sense_type="senses"):
	senses_by_lemmapos = determine_instances_senses(data, counter, sense_type)
	# how many words are used in only one sense?
	#only_one = [k for k in senses_by_lemmapos.keys() if len(set(senses_by_lemmapos[k])) == 1]
	more_than_one = [k for k in senses_by_lemmapos.keys() if len(set(senses_by_lemmapos[k])) > 1]
	avnumsenses = np.average([len(v) for v in senses_by_lemmapos.values()])
	try:
		percentage = len(more_than_one)*100/len(senses_by_lemmapos)
	except ZeroDivisionError:
		print("Could not calculate the percentage")
		percentage = np.nan
	return percentage, len(more_than_one), len(senses_by_lemmapos), avnumsenses


def determine_instances_senses(data, counter, sense_type):
	no_synset_affected_instances = 0
	senses_by_lemmapos = dict()
	ks_to_ignore = []
	for instance in data:
		k = (instance["lemma"], instance["pos"])
		if k in ks_to_ignore:
			continue
		if k not in senses_by_lemmapos:
			senses_by_lemmapos[k] = []
		if sense_type == "senses":
			senseid = instance["sense"]
		elif sense_type == "supersenses":
			if "#" in instance["sense"]: # this means more than one sense is annotated (it happens in manual annotations)
				senses = instance["sense"].split("#")
				lexname_ids = []
				for sense in senses:
					lexname_ids.append(sense.split("%")[1].split(":")[1])
				lexname_ids = sorted(list(set(lexname_ids)))
				if len(lexname_ids) > 1:
					lexname_id = "-".join(lexname_ids)
				else:
					lexname_id = lexname_ids[0]
			else:
				lexname_id = instance["sense"].split("%")[1].split(":")[1]  # info about the format here https://wordnet.princeton.edu/documentation/senseidx5wn
			try:
				senseid = supersense_info[lexname_id]
			except KeyError:
				if "-" in lexname_id:
					senseid = lexname_id
				else:
					pdb.set_trace()
		elif sense_type == "hypernyms":
			# If more than one synset is a hypernym, I sort them alphabetically
			# and the id will be the sense ids, concatenated through "--".
			if "#" in instance["sense"]:
				senses = instance["sense"].split("#")
				all_hypers = []
				for sense in senses:
					hypernyms = wn.lemma_from_key(sense).synset().hypernyms()
					senseid = "--".join(sorted([synset.name() for synset in hypernyms]))
					all_hypers.append(senseid)
				all_hypers = sorted(list(set(all_hypers)))
				if len(all_hypers) > 1:
					senseid = "--".join(all_hypers)
				else:
					senseid = all_hypers[0]
			else:
				try:
					hypernyms = wn.lemma_from_key(instance["sense"]).synset().hypernyms()
					senseid = "--".join([synset.name() for synset in hypernyms])
				except nltk.corpus.reader.wordnet.WordNetError:
					# this happened a few times with hypernyms. These word instances are excluded
					no_synset_affected_instances += 1
					# check how many instances there are of this word. Remove one in the counter. If the counter ends up saying there is 1 or 0 instances left, we ignore it completely
					counter[k] = counter[k] -1
					senseid = None
					if counter[k] < 2:
						ks_to_ignore.append(k)
						senseid = None
						if k in senses_by_lemmapos:
							del senses_by_lemmapos[k]
		else:
			print("SENSE TYPE NOT IMPLEMENTED")
		if senseid is not None:
			senses_by_lemmapos[k].append(senseid)
	print("# of instances affected by wordnet errors:", no_synset_affected_instances)
	return senses_by_lemmapos


def bary_measure(data, counter, sense_type="senses", measure="avgsenses"):

	if measure == "avgsenses":  # this means we dont use wordnet to determine number of senses but the discourse itself
		senses_by_lemmapos = determine_instances_senses(data, counter, sense_type)
		# this is the number of different senses with which a word occurred in this discourse:
		num_senses_by_lemmapos = {(lemma, pos): len(set(senses_by_lemmapos[(lemma, pos)])) for lemma, pos in senses_by_lemmapos}

	elif measure == "pa": # potential polysemy (use wordnet for number of senses)
		all_lemmapos = set([(ins["lemma"],ins["pos"]) for ins in data])
		num_senses_by_lemmapos = {(lemma, pos): find_number_of_senses(lemma, pos, sense_type)[0] for lemma, pos in all_lemmapos}

	denominator = len(data) ## total number of occurrences
	numerator = 0
	numerator_by_number_of_senses = dict()
	for instance in data:
		lemma, pos = instance['lemma'], instance['pos']
		num_of_senses = num_senses_by_lemmapos[(lemma, pos)]
		if num_of_senses not in numerator_by_number_of_senses:
			numerator_by_number_of_senses[num_of_senses] = 0
		numerator_by_number_of_senses[num_of_senses] += 1
	for num_of_senses in numerator_by_number_of_senses:
		numerator += (num_of_senses * numerator_by_number_of_senses[num_of_senses])

	barycenter = numerator / denominator
	numoccs_per_numsense = numerator_by_number_of_senses

	return barycenter, numoccs_per_numsense


def entropy_measure(data, counter, sense_type="senses"):
	senses_by_lemmapos = determine_instances_senses(data, counter, sense_type)
	entropy_by_lemmapos = dict()
	for lemmapos in senses_by_lemmapos.keys():
		# calculate sense distribution. get frequency of each sense.
		senses_counter = Counter(senses_by_lemmapos[lemmapos])
		num_instances = sum(senses_counter.values())
		sense_distribution = {senseid: senses_counter[senseid]/num_instances for senseid in senses_counter}
		entropy = calculate_entropy(list(sense_distribution.values()))
		entropy_by_lemmapos[lemmapos] = entropy

	if len(entropy_by_lemmapos.keys()) == 0:
		entropy_for_discourse = np.nan
	else:
		entropy_for_discourse = sum(entropy_by_lemmapos.values()) / len(entropy_by_lemmapos.keys())

	return entropy_for_discourse




def group_data_by_discourse(dataset_name, poly_words_data):
	if dataset_name == "debates":
		discourse_name = "Topic"
	elif "semeval" in dataset_name or "senseval" in dataset_name:
		discourse_name = "RowID"
	else:
		discourse_name = "Dialogue_ID"
	data_by_discourse = dict()
	for instance in poly_words_data:
		if "semeval" in dataset_name or "senseval" in dataset_name:
			k = instance[discourse_name].split(".")[0]
		else:
			k = instance[discourse_name]
		if k not in data_by_discourse:
			data_by_discourse[k] = []
		data_by_discourse[k].append(instance)
	return data_by_discourse



def calculate_polysemy_measure(poly_words_data, include_adverbs, sense_type="senses", measure="mosd"):
	if measure == "mosd":
		poly_repeated_data, counter = find_repeated_words(poly_words_data, include_adverbs)
		more_than_one_percentage, num_more_than_one, num_lemmapos, avnumsenses = mosd_measure(poly_repeated_data, counter, sense_type=sense_type)
		return more_than_one_percentage, num_more_than_one, num_lemmapos, avnumsenses

	elif measure in ["avgsenses", "pa"]:
		poly_repeated_data, counter = find_repeated_words(poly_words_data, include_adverbs, min_frequency=1)  # for this measure, include words that are not repeated (min_frequency=1)
		barycenter, numoccs_per_numsense = bary_measure(poly_repeated_data, counter, sense_type=sense_type, measure=measure)
		return barycenter, numoccs_per_numsense

	elif "entropy" in measure:
		poly_repeated_data, counter = find_repeated_words(poly_words_data, include_adverbs, min_frequency=2)
		entropy = entropy_measure(poly_repeated_data, counter, sense_type=sense_type)
		return entropy



def load_supersense_info():
	d = dict()
	# key: number, value: supersense name.
	with open("wn_lexnames.csv") as f:
		for i, l in enumerate(f):
			if i > 0:
				l = l.strip().split("\t")
				number, name = l[0], l[1]
				if len(number) == 1:
					number = '0'+ number
				d[number] = name
	return d



def save_results_mosd(num_repeated_poly_lemmapos_all, avnumsenses_all, discourse_onlyone_percentages,
				 discourse_names, out_fn):
	dict_toprint = dict()

	dict_toprint["# repeated poly words"] = num_repeated_poly_lemmapos_all
	dict_toprint["discourse names"] = discourse_names


	dict_toprint["avg # repetitions per repeated poly word"] = avnumsenses_all
	dict_toprint["percentage more than one sense"] = discourse_onlyone_percentages


	df = pd.DataFrame(dict_toprint)
	df.to_csv(path_or_buf=out_fn, sep="\t")



def save_results_bary(barys_by_discourse, numoccs_by_number_of_senses, discourse_names, measure, out_fn):
	dict_toprint = dict()
	dict_toprint["discourse names"] = discourse_names
	dict_toprint["barycenter"] = barys_by_discourse
	all_nums_of_senses = set()
	for dd in numoccs_by_number_of_senses:  # a dictionary corresponding to a discourse
		all_nums_of_senses = all_nums_of_senses.union(set(dd.keys()))
	for num_of_senses in all_nums_of_senses:
		dict_toprint["num_occs_" + str(num_of_senses)] = []
		for disc in numoccs_by_number_of_senses:
			value = disc.get(num_of_senses, 0)
			dict_toprint["num_occs_" + str(num_of_senses)].append(value)

	df = pd.DataFrame(dict_toprint)

	if measure == 'pa':
		pct_poly_data = []
		for i, r in df.iterrows():
			num_mono_instances = r['num_occs_1']
			total_instances = sum([r[k] for k in df.columns if k.startswith("num_occs")])
			pct_poly = (total_instances - num_mono_instances) / total_instances
			pct_poly_data.append(pct_poly)

		df['pct_poly'] = pct_poly_data

	df.to_csv(path_or_buf=out_fn, sep="\t")


def save_results_entropy(entropies_by_discourse, discourse_names, out_fn):
	dict_toprint = dict()
	dict_toprint["discourse names"] = discourse_names
	dict_toprint["entropy"] = entropies_by_discourse

	df = pd.DataFrame(dict_toprint)
	df.to_csv(path_or_buf=out_fn, sep="\t")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_name", default="all", type=str,
						help="In this simplified version, this can be run with 'senseval2' as an example.")
	parser.add_argument("--include_adverbs", action="store_true", help="whether we want to include adverbs in the analysis (not included in the paper)")
	parser.add_argument("--sense_type", default="senses", type=str, help="what kind of senses we use for the analysis:"
						"'senses' for plain wordnet senses, 'supersenses' for wordnet supersenses, 'hypernyms' for hypernyms."
						"This determines 1) what is considered a polysemous word; and 2) when we consider that a word has been used in >1 sense")
	parser.add_argument("--measure", default="avgsenses", type=str,
						help="The name of the polysemy measure to calculate: 'mosd', 'avgsenses', 'pa'(potential ambiguity), 'entropy'")
	parser.add_argument("--output_dir", default="outputs/", type=str,
						help="the name of an existing directory where results will be stored.")
	args = parser.parse_args()

	if args.include_adverbs:
		include_adverbs = True
	else:
		include_adverbs = False

	dataset_names = [args.dataset_name]

	if args.sense_type == "supersenses":
		supersense_info = load_supersense_info()

	for dataset_name in dataset_names:
		# Read annotated senses and merge this information with text data
		print("reading predictions")
		predictions = read_wsd_predictions(dataset_name=dataset_name)
		print("joining predictions with corpus info")
		joint_data, df_data = join_info_and_predictions(dataset_name, predictions, measure=args.measure)

		# Find polysemous words
		poly_words_data = find_polysemous_wordpos(joint_data, sense_type=args.sense_type, measure=args.measure)

		# Organize the data by discourse (i.e. conversation, or text)
		# for the debates: by topic, for the rest: by dialogue id.
		poly_data_by_discourse = group_data_by_discourse(dataset_name, poly_words_data)

		#### Calculating polysemy

		if args.measure == "mosd":
			discourse_onlyone_percentages = []
			num_repeated_poly_lemmapos_all, avnumsenses_all, discourse_names = [], [], []
			for discourse in poly_data_by_discourse:
				more_than_one_percentage, num_more_than_one, num_repeated_poly_lemmapos, avnumsenses = calculate_polysemy_measure(
					poly_data_by_discourse[discourse], include_adverbs=include_adverbs, sense_type=args.sense_type, measure=args.measure)
				discourse_onlyone_percentages.append(more_than_one_percentage)
				num_repeated_poly_lemmapos_all.append(num_repeated_poly_lemmapos)
				avnumsenses_all.append(avnumsenses)
				discourse_names.append(discourse)
		elif args.measure in ["avgsenses", "pa"]:
			barys_per_discourse = []
			nums_occs_by_numsenses = []
			discourse_names = []
			for discourse in poly_data_by_discourse:
				barycenter, num_occs_by_numsenses = calculate_polysemy_measure(
					poly_data_by_discourse[discourse], include_adverbs=include_adverbs, sense_type=args.sense_type, measure=args.measure)
				barys_per_discourse.append(barycenter)
				nums_occs_by_numsenses.append(num_occs_by_numsenses)
				discourse_names.append(discourse)
		elif "entropy" in args.measure:
			entropies_per_discourse = []
			discourse_names = []
			for discourse in poly_data_by_discourse:
				entropy = calculate_polysemy_measure(poly_data_by_discourse[discourse], include_adverbs=include_adverbs, sense_type=args.sense_type,
					measure=args.measure)
				entropies_per_discourse.append(entropy)
				discourse_names.append(discourse)

		out_fn = args.output_dir + dataset_name + "_" + args.sense_type + "_" + args.measure + ".csv"

		if args.measure == "mosd":
			save_results_mosd(num_repeated_poly_lemmapos_all, avnumsenses_all,
								   discourse_onlyone_percentages, discourse_names, out_fn)
		elif args.measure in ["avgsenses", "pa"]:
			save_results_bary(barys_per_discourse, nums_occs_by_numsenses, discourse_names, args.measure, out_fn)
		elif "entropy" in args.measure:
			save_results_entropy(entropies_per_discourse, discourse_names, out_fn)



