import os
import csv
import codecs
import urllib.request

from configs import paths

# Some metadata about ASAP-AES dataset, in order to extract valid items.
set_valid = [str(i) for i in range(1, 9)]
score_valid = [str(i) for i in range(0, 61)]

def load_asap(path=paths.asap, domain_id=None):
    """Read ASAP-AES dataset from Tab Separated Value (TSV) file.
    Note that this function would automatically detect or download ASAP-AES
    dataset.
    Args:
        - path (option): identifies ASAP data file manually.
        - domain_id (option) : identifies the specific prompt of data,
                               everything if not given.
    Returns:
        - Raw ASAP-AES data of OrderedDict, in which "essays", "essay_set" and
          "domain1_score" is crucial.
    """
    if domain_id:
        print("[Loading] ASAP-AES domain {} dataset...".format(domain_id))
    else:
        print("[Loading] ASAP-AES dataset...")
    try:
        with codecs.open(path, "r", "ISO-8859-2") as asap_file:
            asap_reader = csv.DictReader(asap_file, delimiter="\t")
            # Extract valid items in the dataset.
            if not domain_id:
                asap_data = [item for item in asap_reader
                             if item["essay"]
                             and item["essay_set"] in set_valid
                             and item["domain1_score"] in score_valid]
            else:
                asap_data = [item for item in asap_reader
                             if item["essay"]
                             and item["essay_set"] == str(domain_id)
                             and item["domain1_score"] in score_valid]
        return asap_data
    except FileNotFoundError:
        # Auto download.
        if path == paths.asap:
            print("[Downlading] ASAP-AES dataset...")
            urllib.request.urlretrieve(paths.asap_url, path)
            return load_asap(path, domain_id)
        else:
            print("[Error] Seems you identified an non-existing dataset...")

data = load_asap()
print(len(data))
for item in data:
    if item.get("essay_set") == '8': #7 - 8
    #if item.get("essay_id") == '21596':  # 7 - 8
        print(item)




'''
OrderedDict([('essay_id', '21596'), ('essay_set', '8'), ('essay', " I woke up just like any other day happy yet lacking sleep. 
As i got out of bed i would have never known that to day would be the funniest day of my life. I got ready for school after getting 
out of bed. When i got to school every thing seemed like our normal homecoming tell there was a announcement on the intercoms that 
had told every body out of no where there was a dance tonight. So after school was done me and my friends were going to head over 
to our house's to get dressed for the dance. After we were all dressed @PERSON1 picked us all up and we headed to the dance looking fly.
When we got there every body was looking dressed to dance except one guy, he was wearing corduroy pants with a red tucked in flannel
and some brown worn out work boots. We look at him from head to toe and we thought to our self are we in a messed up hillbilly dream? 
That was just the beginning of what was yet to come. As every body started to get in grove of the beat we soon all started dancing to 
the music the music was good and every body was having a good time even the kid with the flannel. But just as every thing was going good 
a song came on that was called cotton eyed @CAPS1 when the flanneled kid heard this song he almost jumped out of his corduroy pants he
soon stared kicking and swinging his feet and arms like if they had no bone or joints in them. Every body started to form a circular
around the kid and every body was laughing and copying the kids movement even us. He didn't rely care he just kept dancing and singing
to the song. The funnest thing about this was that the dance was a formal one and yet this kid manged to pull off wearing a flannel
, some boots, and a pair ofcorduroy pants this kid was out of his mind in fact we still laugh and talk about it tell this day."), 
('rater1_domain1', '15'),
('rater2_domain1', '16'),
('rater3_domain1', ''),
('domain1_score', '31'),
('rater1_domain2', ''),
('rater2_domain2', ''), 
('domain2_score', ''),
('rater1_trait1', '3'),
('rater1_trait2', '3'),
('rater1_trait3', '3'),
('rater1_trait4', '3'),
('rater1_trait5', '3'),
('rater1_trait6', '3'),
('rater2_trait1', '3'),
('rater2_trait2', '3'),
('rater2_trait3', '4'),
('rater2_trait4', '4'),
('rater2_trait5', '4'),
('rater2_trait6', '3'),
('rater3_trait1', ''),
('rater3_trait2', ''),
('rater3_trait3', ''),
('rater3_trait4', ''),
('rater3_trait5', ''),
('rater3_trait6', '')])

'''

