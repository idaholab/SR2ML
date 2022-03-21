import logging
from RuleBasedMatcher import RuleBasedMatcher
import spacy


logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
# To enable the logging to both file and console, the logger for the main should be the root,
# otherwise, a function to add the file handler and stream handler need to be created and called by each module.
# logger = logging.getLogger(__name__)
logger = logging.getLogger()
# # create file handler which logs debug messages
fh = logging.FileHandler(filename='main.log', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

if __name__ == "__main__":
  nlp = spacy.load("en_core_web_sm")
  doc = r"""A leak was noticed from the RCP pump 1A.
          RCP pump 1A pressure gauge was found not operating.
          RCP pump 1A pressure gauge was found inoperative.
          Rupture of pump bearings caused shaft degradation.
          Rupture of pump bearings caused shaft degradation and consequent flow reduction.
          Pump power supply has been found burnout.
          Pump test failed due to power supply failure.
          Pump inspection revealed excessive impeller degradation.
          Pump inspection revealed excessive impeller degradation likely due to cavitation.
        """
  matcher = RuleBasedMatcher(nlp, match=True, phraseMatch=True)
  name = 'ssc_match'
  rules = [{"LOWER":"pump"}, {"POS":"NOUN"}]
  matcher.addPattern(name, rules, callback=None)
  name = 'ssc_phrase_match'
  phraseList = ['pump 1A', 'pump bearings', 'Pump']
  matcher.addPhrase(name, phraseList, callback=None)
  name = 'ssc_dependency_match'
  dependencyList = [
    {
      "RIGHT_ID":"anchor_found",
      "RIGHT_ATTRS":{"ORTH":"found"}
    },
    {
      "LEFT_ID":"anchor_found",
      "REL_OP":">",
      "RIGHT_ID":"subject",
      "RIGHT_ATTRS":{"DEP":"nsubjpass"}
    },
    {
      "LEFT_ID":"anchor_found",
      "REL_OP":">",
      "RIGHT_ID":"object",
      "RIGHT_ATTRS":{"DEP":"oprd"}
    }
  ]
  matcher.addDependency(name, dependencyList, callback=None)
  name = 'ssc_entity_ruler'
  patterns = [{"label":"pump_comp", "pattern":[{"LOWER":"pump"}, {"POS":"NOUN"}], "id":"ssc"}]
  matcher.addEntityPattern(name, patterns)

  matcher(doc)
