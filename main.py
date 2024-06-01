from openai import OpenAI 
import os
from dotenv import load_dotenv
import re
import json
import networkx as nx
import matplotlib.pyplot as plt

load_dotenv()
api_key=os.getenv("OPENAI_KEY")

MODEL="gpt-4o"
client = OpenAI(api_key=api_key)

# Custom text chunk
text_chunk = """Yann André LeCun[1] (/ləˈkʌn/ lə-KUN, French: [ləkœ̃];[2] originally spelled Le Cun;[2] born 8 July 1960) is a French-American computer scientist working primarily in the fields of machine learning, computer vision, mobile robotics and computational neuroscience. He is the Silver Professor of the Courant Institute of Mathematical Sciences at New York University and Vice-President, Chief AI Scientist at Meta.[3][4]

He is well known for his work on optical character recognition and computer vision using convolutional neural networks (CNNs).[5][6] He is also one of the main creators of the DjVu image compression technology (together with Léon Bottou and Patrick Haffner). He co-developed the Lush programming language with Léon Bottou.

LeCun received the 2018 Turing Award, together with Yoshua Bengio and Geoffrey Hinton, for their work on deep learning.[7] The three are sometimes referred to as the "Godfathers of AI" and "Godfathers of Deep Learning".[8][9][10][11][12][13]

Early life and education

LeCun at the University of Minnesota, 2014
LeCun was born on 8 July 1960, at Soisy-sous-Montmorency in the suburbs of Paris. His name was originally spelled Le Cun from the old Breton form Le Cunff and was from the region of Guingamp in northern Brittany. "Yann" is the Breton form for "John".[2]

He received a Diplôme d'Ingénieur from the ESIEE Paris in 1983 and a PhD in Computer Science from Université Pierre et Marie Curie (today Sorbonne University) in 1987 during which he proposed an early form of the back-propagation learning algorithm for neural networks.[14]

Career
Bell Labs
In 1988, LeCun joined the Adaptive Systems Research Department at AT&T Bell Laboratories in Holmdel, New Jersey, United States, headed by Lawrence D. Jackel, where he developed a number of new machine learning methods, such as a biologically inspired model of image recognition called convolutional neural networks,[15] the "Optimal Brain Damage" regularisation methods,[16] and the Graph Transformer Networks method (similar to conditional random field), which he applied to handwriting recognition and OCR.[17] The bank check recognition system that he helped develop was widely deployed by NCR and other companies, reading over 10% of all the checks in the US in the late 1990s and early 2000s.[citation needed]

In 1996, he joined AT&T Labs-Research as head of the Image Processing Research Department, which was part of Lawrence Rabiner's Speech and Image Processing Research Lab, and worked primarily on the DjVu image compression technology,[18] used by many websites, notably the Internet Archive, to distribute scanned documents.[citation needed] His collaborators at AT&T include Léon Bottou and Vladimir Vapnik.

New York University
After a brief tenure as a Fellow of the NEC Research Institute (now NEC-Labs America) in Princeton, NJ, LeCun joined New York University (NYU) in 2003, where he is Jacob T. Schwartz Chaired Professor of Computer Science and Neural Science at the Courant Institute of Mathematical Sciences and the Center for Neural Science. He is also a professor at the Tandon School of Engineering.[19][20] At NYU, he has worked primarily on Energy-Based Models for supervised and unsupervised learning,[21] feature learning for object recognition in Computer Vision,[22] and mobile robotics.[23]

In 2012, he became the founding director of the NYU Center for Data Science.[24] On 9 December 2013, LeCun became the first director of Meta AI Research in New York City,[25][non-primary source needed][26] and stepped down from the NYU-CDS directorship in early 2014.

In 2013, he and Yoshua Bengio co-founded the International Conference on Learning Representations, which adopted a post-publication open review process he previously advocated on his website. He was the chair and organiser of the "Learning Workshop" held every year between 1986 and 2012 in Snowbird, Utah. He is a member of the Science Advisory Board of the Institute for Pure and Applied Mathematics[27] at UCLA. He is the Co-Director of the Learning in Machines and Brain research program (formerly Neural Computation & Adaptive Perception) of CIFAR.[28]

In 2016, he was the visiting professor of computer science on the "Chaire Annuelle Informatique et Sciences Numériques" at Collège de France in Paris, where he presented the "leçon inaugurale" (inaugural lecture).[29] In 2023, he was named as the inaugural Jacob T. Schwartz Chaired Professor in Computer Science at NYU's Courant Institute.[30]

Honours and awards
LeCun is a member of the US National Academy of Sciences,[31] National Academy of Engineering and the French Académie des Sciences.

He has received honorary doctorates from IPN in Mexico City[32] in 2016, from EPFL[33][34] in 2018, from Université Côte d'Azur in 2021,[35] from Università di Siena in 2023,[36] and from Hong Kong University of Science and Technology in 2023.

In 2014, he received the IEEE Neural Network Pioneer Award and in 2015, the PAMI Distinguished Researcher Award.[37]

In 2018, LeCun was awarded the IRI Medal, established by the Industrial Research Institute (IRI),[38] and the Harold Pender Award, given by the University of Pennsylvania,[39]

In 2019, he received the Golden Plate Award of the American Academy of Achievement.[40]

In 2022, he received the Princess of Asturias Award in the category "Scientific Research", along with Yoshua Bengio, Geoffrey Hinton and Demis Hassabis.[41]

In 2023, the President of France made him a Chevalier (Knight) of the French Legion of Honour.[42]

During the World Economic Forum (WEF) 2024 in Davos, he received the Global Swiss AI Award 2023.[43]

Turing Award
In March 2019, LeCun won the 2018 Turing award, sharing it with Yoshua Bengio and Geoffrey Hinton.[44]"""

# 1 - Basic Chat

completion = client.chat.completions.create(
  model=MODEL,
  messages=[
    {"role": "system", "content": 'You are an expert Knowledge Graph creator. For the given text chunks, identify the entities, nodes, and relationship triplets. It should be in the JSON format with keys [{"Subject":"", "Object":"", "Relation":""}, {"Subject":"", "Object":"", "Relation":""}, ...]. Find all triplet pairs present in it.'},
    {"role": "user", "content": text_chunk}
  ]
)

text=completion.choices[0].message.content
print(type(text))
print("Assistant: " + completion.choices[0].message.content)

def extract_json_content(text):
    # Regex pattern to match JSON content
    pattern = r'```json\s*(\[.*?\])\s*```'
    
    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)
    
    # If a match is found, return the JSON content
    if match:
        return match.group(1)
    else:
        return None

clean_text = extract_json_content(text)

# Load JSON content
if clean_text:
    json_data = json.loads(clean_text)
    print(json_data)
else:
    print("No JSON content found.")

G = nx.DiGraph()

for triplet in json_data:
    subject = triplet["Subject"]
    object_ = triplet["Object"]
    relation = triplet["Relation"]
    G.add_node(subject)
    G.add_node(object_)
    G.add_edge(subject, object_, label=relation)

pos = nx.spring_layout(G)
labels = nx.get_edge_attributes(G, 'label')

plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
plt.title("Knowledge Graph")
plt.show()