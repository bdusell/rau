def add_tag(model, tag: str | None):
    if tag is not None:
        return model.tag(tag)
    else:
        return model.main()
