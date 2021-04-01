from typing import List
from core.training.training_plugin_manager import TrainingPluginManager, LOG_PREPROCESSOR_CATEGORY, LOG_TEMPLATE_GENERATOR_CATEGORY, \
    LOG_WORD_EMBEDDER_CATEGORY
from core.training.abstract.word_embeddings import LogPreprocessor, TemplateGenerator, WordEmbedder

PLUGIN_MANAGER = TrainingPluginManager()

LOG_PREPROCESSOR_CATEGORY: List[LogPreprocessor] = \
    PLUGIN_MANAGER.get_plugins_from_category(LOG_PREPROCESSOR_CATEGORY)
LOG_TEMPLATE_GENERATOR_CATEGORY: List[TemplateGenerator] = \
    PLUGIN_MANAGER.get_plugins_from_category(LOG_TEMPLATE_GENERATOR_CATEGORY)
LOG_WORD_EMBEDDER_CATEGORY: List[WordEmbedder] = \
    PLUGIN_MANAGER.get_plugins_from_category(LOG_WORD_EMBEDDER_CATEGORY)