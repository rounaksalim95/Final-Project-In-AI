import time
import sys
import os
import urllib2


########### keywrod & spec ###########


search_keyword = ['banana']

# narrow the search keyword
keywords = [' ']

########### End of Editing ###########


# downloading entire Web Document
def download_page(url):
    version = (3,0)
    cur_version = sys.version_info
    if cur_version >= version:  # python 3.x
        import urllib.request
        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
            req = urllib.request.Request(url, headers = headers)
            resp = urllib.request.urlopen(req)
            respData = str(resp.read())
            return respData
        except Exception as e:
            print(str(e))
    else:  # python 2.x
        import urllib2
        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"
            req = urllib2.Request(url, headers = headers)
            response = urllib2.urlopen(req)
            page = response.read()
            return page
        except:
            return"Page Not found"


# finding 'Next Image' from the given raw page
def _images_get_next_item(s):
    start_line = s.find('rg_di')
    if start_line == -1:    #If no links are found then give an error
        end_quote = 0
        link = "no_links"
        return link, end_quote
    else:
        start_line = s.find('"class="rg_meta"')
        start_content = s.find('"ou"',start_line+1)
        end_content = s.find(',"ow"',start_content+1)
        content_raw = str(s[start_content+6:end_content-1])
        return content_raw, end_content


# getting all links
def _images_get_all_items(page):
    items = []
    while True:
        item, end_content = _images_get_next_item(page)
        if item == "no_links":
            break
        else:
            items.append(item)
            time.sleep(0.05)
            page = page[end_content:]
    return items



############## Main Program ############
t0 = time.time()   #start the timer

# download image links
i= 0
while i<len(search_keyword):
    items = []
    iteration = "Item no.: " + str(i+1) + " -->" + " Item name = " + str(search_keyword[i])
    print (iteration)
    print ("Evaluating...")
    search_keywords = search_keyword[i]
    search = search_keywords.replace(' ','%20')
    j = 0
    while j<len(keywords):
        pure_keyword = keywords[j].replace(' ','%20')
        url = 'https://www.google.com/search?q=' + search + pure_keyword + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
        raw_html =  (download_page(url))
        time.sleep(0.1)
        items = items + (_images_get_all_items(raw_html))
        j = j + 1
    print ("Total Image Links = "+str(len(items)))
    print ("\n")
    i = i+1

    info = open('output.txt', 'a')
    info.write(str(i) + ': ' + str(search_keyword[i-1]) + ": " + str(items) + "\n\n\n")
    info.close()

t1 = time.time()
total_time = t1-t0
print("Total time taken: "+str(total_time)+" Seconds")
print ("Starting Download...")

# start downloading images
DIR="Pictures"
if not os.path.exists(DIR):
    os.mkdir(DIR)
DIR = os.path.join(DIR, search_keyword[0])
if not os.path.exists(DIR):
    os.mkdir(DIR)

k=0
errorCount=0
while(k<len(items)):
    from urllib2 import Request,urlopen
    from urllib2 import URLError, HTTPError

    try:
        req = Request(items[k], headers={"User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
        response = urlopen(req)
        output_file = open(os.path.join(DIR , str(k+100)+".jpg"), 'wb')
        data = response.read()
        output_file.write(data)
        response.close();
        print("completed ====> "+str(k+100))
        k=k+1;
    except IOError:
        errorCount+=1
        print("IOError on image "+str(k+100))
        k=k+1;
    except HTTPError as e:
        errorCount+=1
        print("HTTPError"+str(k))
        k=k+1;
    except URLError as e:
        errorCount+=1
        print("URLError "+str(k))
        k=k+1;

print("\n")
print("All are downloaded")
print("\n"+str(errorCount)+" ----> total Errors")
