def add_tag(model, tag):
    if tag is None:
        return model.main()
    else:
        return model.tag(tag)
