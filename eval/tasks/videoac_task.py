import os
from lmms_eval.api.task import ConfigurableTask
from lmms_eval.api.registry import register_task
from lmms_eval.filters import build_filter_ensemble
from . import utils

@register_task("videoac_switch")
class VideoAC(ConfigurableTask):

    def __init__(self, config):
        # Initialize without calling parent's __init__ to avoid download
        self._config = config
        self.DATASET_PATH = config.dataset_path
        self._training_docs = None
        self._fewshot_docs = None
        self._instances = None
        self._filters = [build_filter_ensemble("none", [["take_first", None]])]
        
        # Set required attributes that ConfigurableTask expects
        self.model_name = None
        self.OUTPUT_TYPE = "generate_until"
        self.VERSION = "Custom"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        if not hasattr(self, '_validation_docs'):
            self._validation_docs = utils.videoac_get_documents(self.DATASET_PATH)
        return self._validation_docs

    def download(self, dataset_kwargs):
        """Override download to avoid calling parent class download"""
        # Do nothing - we'll load data in validation_docs
        pass

    def doc_to_visual(self, doc):
        return utils.videoac_doc_to_visual(doc)

    def doc_to_text(self, doc):
        # The lmms_eval_specific_kwargs are passed to the Task constructor
        # and stored in self.config
        return utils.videoac_doc_to_text(doc, lmms_eval_specific_kwargs=self.config)

    def process_results(self, doc, results):
        return utils.videoac_process_results(doc, results)

    def higher_is_better(self):
        return {"accuracy": True}

    def aggregation(self):
        return {"accuracy": utils.videoac_accuracy}

    def doc_to_target(self, doc):
        """Return the target for the document"""
        return doc.get("answer", "")

    def fewshot_docs(self):
        """Return fewshot examples if available"""
        return []

    def construct_requests(self, doc, ctx):
        """Construct requests for the task"""
        # This is a simple implementation for generate_until tasks
        return [doc]
