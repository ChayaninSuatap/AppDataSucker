class CrawlState :
    state = {}

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
