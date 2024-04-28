from .abstract_feature_function import AbstractFeatureFunction

class IdentityFeature(AbstractFeatureFunction):
    def __call__(self, value):
        super(IdentityFeature, self).__call__(value)
        return value.float()
