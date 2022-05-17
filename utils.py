import pandas as pd
import xml.etree.ElementTree as ET

def read_wsd_predictions(dataset_name):
	'''
	:param fn: str, filename of the file containing wsd predictions
	:returns predictions: list of (instance_id, sense_id) tuples
	'''
	annotation_dir = "Data/"
	if "-gold" in dataset_name:
		dn = dataset_name.split("-")[0]
		fn = "Data/" + dn + ".gold.key.txt"
	else:
		fn = annotation_dir + dataset_name + "_predictions.txt"
	predictions = list(map(str.split, list(map(str.strip, open(fn).readlines()))))
	return predictions



def open_xml(dataset_name):
	if "-gold" in dataset_name:
		dataset_name = dataset_name.split("-")[0]
	xmlfn = "Data/" + dataset_name + ".data.xml"
	print("parsing xml")
	xml_tree = ET.parse(xmlfn)
	root = xml_tree.getroot()
	return root


def load_and_prepare_dataset(dataset_name):
	''' Turn the xml file into a df for convenience'''
	if "semeval" in dataset_name or "senseval" in dataset_name:
		data = create_df_from_semeval_xml(dataset_name)
	else:
		print("NOT IMPLEMENTED: The function load_and_prepare_dataset needs to be adapted for non-semeval datasets")

	df = pd.DataFrame(data)
	return df


def create_df_from_semeval_xml(dataset_name):
	fn = "Data/" + \
		 dataset_name.split("-")[0] + ".data.xml"
	dict_to_df = []
	tree = ET.parse(fn)
	root = tree.getroot()
	for text_xml in root:
		for sentence_xml in text_xml:
			sentence_txt = []
			for word in sentence_xml:
				sentence_txt.append(word.text)
			sentence_dict = {"RowID": sentence_xml.attrib['id'], "Utterance": " ".join(sentence_txt)}
			dict_to_df.append(sentence_dict)
	return dict_to_df



def posmap_u_to_wn(pos):
	posmap = dict({"NOUN":"n","VERB":"v","ADJ":"a","ADV":"r"})
	return posmap.get(pos, pos)


def determine_text_id_from_sentence_id(sentence_id, dataset_name):
	if "semeval" in dataset_name or "senseval" in dataset_name:
		text_id = ".".join(sentence_id.split(".")[:2])
	else:
		text_id = sentence_id.split(".")[0]
	if dataset_name == "debates":
		text_id = int(text_id)
	return text_id

