import mypath

class CrawlState :
    state = {}

    def __init__(self) :
        self.load_state()
    
    def save_state(self):
        fs = open( mypath.crawlstate_txt, 'w')
        for k,v in self.state.items() :
            fs.write(str(v) +  k + '\n')
        fs.close()
    
    def load_state(self):
        try:
            fs = open( mypath.crawlstate_txt)
            for line in fs :
                v = int(line[0])
                k = line[1:-1]
                self.state[k] = v
        except:
            pass

    def add(self, link):
        if link not in self.state :
            self.state[link] = 0
    
    def add_links(self, links):
        for link in links :
            if link not in self.state :
                self.state[link] = 0
    
    def get_uncrawled_links(self):
        output = []
        for link in self.state :
            if self.state[link] == 0 :
                output.append(link)
        return output

    def mark_as_crawled(self, link):
        if link in self.state :
            self.state[link] = 1

    def has_uncrawled_link(self):
        for link in self.state :
            if self.state[link] == 0 :
                return True
        return False
