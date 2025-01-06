# Polysemy in Spoken Conversations 🗣️ and Written Texts 📄

This repository contains code for our upcoming paper:

Aina Garí Soler, Matthieu Labeau and Chloé Clavel (2022). Polysemy in Spoken Conversations and Written Texts. To appear in _Proceedings of the 13th Language Resources and Evaluation Conference (LREC)_, Marseille, France, June 20-25.


## About the datasets

We cannot directly share the datasets used in the paper nor the annotations derived. However, to test the code, you can download the Senseval and SemEval datasets, which are ready to be used with our code.

You can find the full WSD Evaluation framework (Raganato et al., 2017) [here](http://lcl.uniroma1.it/wsdeval/).

You can use, for example:
* ``Evaluation_Datasets/senseval2/senseval2.gold.key.txt``, containing gold annotations for Senseval2. 
* ``Evaluation_Datasets/senseval2/senseval2.data.xml``, which contains this dataset in xml format. 

Place these two files **under the ``Data/`` directory.**

I plan to add here information on how the other datasets used in the paper were retrieved and turned into xml format before being disambiguated. In the meantime feel free to [contact me](#contact). 

We do include the utterance IDs of the debates which served as monologs. You can find them in `Data/debate_monologue_ids.txt`. The other IDs (only those that correspond to Trump's or Biden's utterances) counted as dialog utterances and were split by topic. This dataset can be found [here](https://www.kaggle.com/datasets/rmphilly18/us-presidential-debatefinal-october-2020).


## Automatic WSD

Automatic WSD annotation was performed with the ESCHER model (Barba et al., 2021) with the code available [in their repository](https://github.com/SapienzaNLP/esc). 


## Calculating polysemy measures

In this repository you can find a simplified version of the original code which can serve to calculate the 5 polysemy measures described in the paper for a text annotated with senses. See the section above to obtain example input files. Then run:

``read_wsd_predictions.py --dataset_name senseval2-gold --sense_type senses --measure mosd --output_dir outputs/``

More information on the arguments:
* ``--dataset_name``
* ``--include_adverbs`` if indicated, we include adverbs in the polysemy counts. By default, we do not.
* ``--sense_type`` can be _senses_, _supersenses_ or _hypernyms_.	
* ``--measure`` can be _mosd_, _avgsenses_, _pa_ or _entropy_. When using _pa_ (Potential Ambiguity), the output for the measure PCT-POLY is automatically included in the output file.
* ``--output_dir`` the name of an existing directory where results will be stored.

The output is printed to a csv file.

To use the code with your own dataset, you should have two files in ``Data/``:

* A file containing WSD annotations (the output of the WSD model). The filename should be ``{DATASETNAME}_predictions.txt`` (where ``{DATASETNAME}`` is the name to be used for the ``--dataset_name`` argument of ``read_wsd_predictions.py``.
* A file containing the text content in xml format. The filename should be ``{DATASETNAME}.data.xml``.

You will need to modify the function ``load_and_prepare_dataset`` in ``utils.py`` to convert your xml into a pandas DataFrame. Let me know if you need help with that step.


### Citation

If you use the code in this repository, please cite our paper:

```
@inproceedings{gari-soler-etal-2022-polysemy,
    title = {{Polysemy in Spoken Conversations and Written Texts}},
    author = "Gar{\'i} Soler, Aina  and
      Labeau, Matthieu  and
      Clavel, Chlo{\'e}",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.179/",
    pages = "1680--1690"
}

    
```

### References

* Raganato, A., Camacho-Collados, J., and Navigli, R. (2017). [Word Sense Disambiguation: A Unified Evaluation Framework and Empirical Comparison.](https://aclanthology.org/E17-1010.pdf) In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pages 99–110,
Valencia, Spain, April. Association for Computational Linguistics.

* Barba, E., Pasini, T., and Navigli, R. (2021). [ESC: Redesigning WSD with Extractive Sense Comprehension](https://aclanthology.org/2021.naacl-main.371/). In Proceedings of the 2021 Conference of
the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4661–4672, Online, June. Association for Computational Linguistics



### Contact <a name="contact"></a>

Feel free to [contact me](https://ainagari.github.io/menu/contact.html) for any questions or requests.
