import os
import plugins

from yapsy.PluginManager import PluginManager
from core.training.abstract.word_embeddings import LogPreprocessor, TemplateGenerator, WordEmbedder


LOG_PREPROCESSOR_CATEGORY = 'LogPreprocessor'
LOG_TEMPLATE_GENERATOR_CATEGORY = 'TemplateGenerator'
LOG_WORD_EMBEDDER_CATEGORY = 'WordEmbedder'


class TrainingPluginManager:

    def __init__(self):
        self.manager = PluginManager(
            categories_filter={
                LOG_PREPROCESSOR_CATEGORY: LogPreprocessor,
                LOG_TEMPLATE_GENERATOR_CATEGORY: TemplateGenerator,
                LOG_WORD_EMBEDDER_CATEGORY: WordEmbedder
            },
            directories_list=[os.path.dirname(plugins.__file__)]
        )
        self.manager.collectPlugins()

    def get_plugins_from_category(self, category_filter):
        return [plugin.plugin_object for plugin in self.manager.getPluginsOfCategory(category_filter)]
