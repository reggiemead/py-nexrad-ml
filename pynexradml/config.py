import ConfigParser

class NexradConfig(object):
    def __init__(self, fileName, section):
        self.parser = ConfigParser.ConfigParser()
        self.parser.read(fileName)
        self.items = {}
        if self.parser.has_section(section):
            self.items = dict(self.parser.items(section))

    def getOverrideOrConfig(self, args, item):
        if hasattr(args, item) and getattr(args, item) != None:
            return getattr(args, item)
        elif item in self.items:
            return self.items[item]
        else:
            return None

    def getOverrideOrConfigAsList(self, args, item):
        if hasattr(args, item) and getattr(args, item) != None and len(getattr(args, item)) > 0:
            return getattr(args, item)
        elif item in self.items:
            return self.items[item].split()
        else:
            return None

    def getOverrideOrConfigAsBool(self, args, item):
        if hasattr(args, item) and getattr(args, item) != None:
            return getattr(args, item)
        elif item in self.items:
            result = self.items[item]
            return result and result != "False" and result != "false"
        else:
            return None

    def getOverrideOrConfigAsFloat(self, args, item):
        if hasattr(args, item) and getattr(args, item) != None:
            return getattr(args, item)
        elif item in self.items:
            return float(self.items[item])
        else:
            return None

    def getOverrideOrConfigAsInt(self, args, item):
        if hasattr(args, item) and getattr(args, item) != None:
            return getattr(args, item)
        elif item in self.items:
            return int(self.items[item])
        else:
            return None
