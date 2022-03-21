import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Token

# We're using a component factory because the component needs to be
# initialized with the shared vocab via the nlp object
@Language.factory("html_merger")
def create_bad_html_merger(nlp, name):
    return BadHTMLMerger(nlp.vocab)

class BadHTMLMerger:
    def __init__(self, vocab):
        patterns = [
            [{"ORTH": "<"}, {"LOWER": "br"}, {"ORTH": ">"}],
            [{"ORTH": "<"}, {"LOWER": "br/"}, {"ORTH": ">"}],
        ]
        # Register a new token extension to flag bad HTML
        Token.set_extension("bad_html", default=False)
        self.matcher = Matcher(vocab)
        self.matcher.add("BAD_HTML", patterns)

    def __call__(self, doc):
        # This method is invoked when the component is called on a Doc
        matches = self.matcher(doc)
        spans = []  # Collect the matched spans here
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)
                for token in span:
                    token._.bad_html = True  # Mark token as bad HTML
        return doc

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("html_merger", last=True)  # Add component to the pipeline
doc = nlp("Hello<br>world! <br/> This is a test.")
for token in doc:
    print(token.text, token._.bad_html)

## Take a path to a JSON file containing the patterns

# @Language.factory("html_merger", default_config={"path": None})
# def create_bad_html_merger(nlp, name, path):
#     return BadHTMLMerger(nlp, path=path)
#
# nlp.add_pipe("html_merger", config={"path": "/path/to/patterns.json"})

### Examples for customer pipeline
from spacy.language import Language

# Usage as a decorator
@Language.component("my_component")
def my_component(doc):
   # Do something to the doc
   return doc

# Usage as a function
Language.component("my_component2", func=my_component)

###############
from spacy.language import Language

# Usage as a decorator
@Language.factory(
   "my_component",
   default_config={"some_setting": True},
)
def create_my_component(nlp, name, some_setting):
     return MyComponent(some_setting)

# Usage as function
Language.factory(
    "my_component",
    default_config={"some_setting": True},
    func=create_my_component
)

##############################################################
import spacy
from spacy.language import Language
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")

@Language.component("expand_person_entities")
def expand_person_entities(doc):
    new_ents = []
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.start != 0:
            prev_token = doc[ent.start - 1]
            if prev_token.text in ("Dr", "Dr.", "Mr", "Mr.", "Ms", "Ms."):
                new_ent = Span(doc, ent.start - 1, ent.end, label=ent.label)
                new_ents.append(new_ent)
        else:
            new_ents.append(ent)
    doc.ents = new_ents
    return doc

# Add the component after the named entity recognizer
nlp.add_pipe("expand_person_entities", after="ner")

doc = nlp("Dr. Alex Smith chaired first board meeting of Acme Corp Inc.")
print([(ent.text, ent.label_) for ent in doc.ents])

#### Another way is to use Span set_extension to add the custom extension attribute.
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")

def get_person_title(span):
    if span.label_ == "PERSON" and span.start != 0:
        prev_token = span.doc[span.start - 1]
        if prev_token.text in ("Dr", "Dr.", "Mr", "Mr.", "Ms", "Ms."):
            return prev_token.text

# Register the Span extension as 'person_title'
Span.set_extension("person_title", getter=get_person_title)

doc = nlp("Dr Alex Smith chaired first board meeting of Acme Corp Inc.")
print([(ent.text, ent.label_, ent._.person_title) for ent in doc.ents])




###### Can be used for relationship extraction:
import spacy
from spacy.language import Language
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

# @Language.component("extract_person_orgs")
# def extract_person_orgs(doc):
#     person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]
#     for ent in person_entities:
#         head = ent.root.head
#         if head.lemma_ == "work":
#             preps = [token for token in head.children if token.dep_ == "prep"]
#             for prep in preps:
#                 orgs = [token for token in prep.children if token.ent_type_ == "ORG"]
#                 print({'person': ent, 'orgs': orgs, 'past': head.tag_ == "VBD"})
#     return doc

@Language.component("extract_person_orgs")
def extract_person_orgs(doc):
    person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    for ent in person_entities:
        head = ent.root.head
        if head.lemma_ == "work":
            preps = [token for token in head.children if token.dep_ == "prep"]
            for prep in preps:
                orgs = [t for t in prep.children if t.ent_type_ == "ORG"]
                aux = [token for token in head.children if token.dep_ == "aux"]
                past_aux = any(t.tag_ == "VBD" for t in aux)
                past = head.tag_ == "VBD" or head.tag_ == "VBG" and past_aux
                print({'person': ent, 'orgs': orgs, 'past': past})
    return doc

# To make the entities easier to work with, we'll merge them into single tokens
nlp.add_pipe("merge_entities")
nlp.add_pipe("extract_person_orgs")

doc = nlp("Alex Smith worked at Acme Corp Inc.")
# If you're not in a Jupyter / IPython environment, use displacy.serve
displacy.render(doc, options={"fine_grained": True})
