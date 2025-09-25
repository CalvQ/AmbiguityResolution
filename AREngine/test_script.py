from arengine import AREngine

arengine = AREngine()
arengine.load_model()

arengine.resolve_prompt("This is not ambiguous")
arengine.resolve_prompt("This is ambiguous")