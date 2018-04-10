#coding=utf-8
import lxml.html
import pprint
import urllib2
import urllib
import cookielib

def parse_form(html):
    tree = lxml.html.fromstring(html)
    data={}
    for e in tree.cssselect('form input'):
        if e.get('name'):
            data[e.get('name')] = e.get('value')
    return data

if __name__  == '__main__':
    cj = cookielib.CookieJar()
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))


    Login_url = 'https://github.com/login'
    login_email = 'mambasmile'
    login_passwd = '-------'

    html = opener.open(Login_url).read()
    print html

    data = parse_form(html)

    data['login']=login_email
    data['password'] = login_passwd

    tmp_data={}
    for k, v in data.iteritems():
        tmp_data[k] = unicode(v).encode('utf-8')

    print tmp_data
    encoded_data = urllib.urlencode(tmp_data)
    
    headers = {
        "Host":"assets-cdn.github.com",
        "User-Agent":"Mozilla/5.0 (X11; Ubuntu; Lin… Gecko/20100101 Firefox/54.0"
    }
    request = urllib2.Request(Login_url,encoded_data)
    response = opener.open(request)
    print response.geturl()
