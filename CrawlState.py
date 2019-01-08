import mypath
import link_category_util

class CrawlState() :
    state = {}
    save_interval_tick = 0
    save_interval = 0

    def __init__(self, save_interval) :
        self.load_state()
        self.save_interval = save_interval
    
    def save_state(self):
        self.save_interval_tick += 1
        if self.save_interval_tick % self.save_interval == 0 :
            self.save_interval_tick = 0
            self.force_save_state()
    
    def force_save_state(self):
        fs = open( mypath.crawlstate_txt, 'w')
        for k,v in self.state.items() :
            fs.write(str(v) +  k + '\n')
        fs.close()
        print('crawl state saved')
    
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
    
    def get_uncrawled_links_no_cluster(self):
        output = []
        for link in self.state :
            if self.state[link] == 0 and not link_category_util.link_is_cluster(link) :
                output.append(link)
        return output 

    def mark_as_crawled(self, link):
        if link in self.state :
            self.state[link] = 1
    
    def unmark_as_crawled(self, link):
        if link in self.state :
            self.state[link] = 0
 

    def has_uncrawled_link(self):
        for link in self.state :
            if self.state[link] == 0 :
                return True
        return False
    
    def has_uncrawled_link_no_cluster(self):
        for link in self.state :
            if self.state[link] == 0 and not link_category_util.link_is_cluster(link) :
                return True
        return False
