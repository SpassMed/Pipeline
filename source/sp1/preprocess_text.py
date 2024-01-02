import re

import requests

def get_doi_lst(txt):
    exp = "10.\\d{4,9}/[-._;()/:a-z0-9A-Z]+"
    pattern = re.compile(exp)
    return pattern.findall(txt)

txt = ['doi:10.1097/BRS.0b013e31829ff095',
'https://dx.doi.org/10.1016/j.arth.2005.04.023',
'https://dx.doi.org/10.1016/j.arth.2005.04.023',
'https://dx.doi.org/10.1186/s13018-017-0552-9',
]

# Example
# txt = "[1] 10.1038/s41598-022-09712-w [2] 10.3390/tropicalmed7110342 [3] 10.3390/life11080790 [4] 10.1155/2022/9367873"
# print(pattern.findall(txt))

# txt2 = "\n[1] 10.1038/s41598-022-09712-w\n[2] 10.3390/tropicalmed7110342\n[3] 10.3390/life11080790\n[4] 10.1155/2022/9367873"
# print(pattern.findall(txt2))

# txt3 = "\n- [1] doi: 10.1038/s41598-022-09712-w\n- [2] doi: 10.3390/tropicalmed7110342\n- [3] doi: 10.3390/life11080790\n- [4] doi: 10.1155/2022/9367873\n- [5] doi: 10.3389/fmed.2022.997992\n"
# print(pattern.findall(txt3))

# txt4 = "[1]. Tropical Medicine and Infectious Disease, 10.3390/tropicalmed7110342 [^1^]\n\n[2]. Scientific Reports, 10.1038/s41598-022-09712-w [^2^]\n\n[3]. Journal of Healthcare Engineering, 10.1155/2022/9367873 [^3^]\n\n[4]. Life (Basel), 10.3390/life11080790 [^4^]\n\n[5]. Canadian Journal of Infectious Diseases and Medical Microbiology, 10.1155/2022/3578528 [^5^]\n   \n[^1^]: https://doi.org/10.3390/tropicalmed7110342\n[^2^]: https://doi.org/10.1038/s41598-022-09712-w\n[^3^]: https://doi.org/10.1155/2022/9367873\n[^4^]: https://doi.org/10.3390/life11080790\n[^5^]: https://doi.org/10.1155/2022/3578528\n"
# print(pattern.findall(txt4))

def get_doi_title(doi):
  base_url = f"https://doi.org/{doi}"
  headers = {
      "Accept": "text/bibliography; style=bibtex"
  }
  response = requests.get(base_url, headers=headers)

  if response.status_code == 200:
    title = response.text.strip()
    _,title =title.split('title={')
    title= title.split('}')[0]
    return title
  else:
    print(response.status_code)
    print(response.text)
    return None

def get_reference(txt):
    doi_lst = get_doi_lst(txt)
    title_lst = []
    refertxt = "References**\n "
    for i in doi_lst:
       title_lst.append(get_doi_title(i))
    for i in range(len(doi_lst)):
       refertxt += f"- [{i+1}] {title_lst[i]}: https://doi.org/{doi_lst[i]}\n "
    return refertxt
       
# Example
# intxt = "*\n- [1] 10.1038/s41598-022-09712-w\n- [2] 10.3390/tropicalmed7110342\n- [3] 10.3390/life11080790\n- [4] 10.1155/2022/9367873\n- [5] 10.3389/fmed.2022.997992"
# print(get_reference(intxt))
       
       
      