import coremltools as ct
model = ct.models.MLModel("./coreml_models.mlpackage")
print("User metadata:", model.user_defined_metadata)
print("All metadata:", model.short_description)