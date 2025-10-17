# Install Dependencies

```bash
pip install -r requirements.txt
```

# Tokenization process

To get tokenization training data as a .txt file, follow these steps:
1. Visit the [Wikipedia English Language Dump](https://dumps.wikimedia.org/enwiki/)
2. Click on [Latest](https://dumps.wikimedia.org/enwiki/latest/)
3. Use **Cmd+F** to find the file at the following date and time: **292240103**
4. Click on **"enwiki-latest-pages-articles-multistream1.xml-p..>"** to download the file
5. Once in finder system, the file is in a .bz2 format. Double click the file to expand it into a .xml file.
6. Add this to the folder **'tokenizer_training_data'**
7. Create a file called **'all_wiki_text.txt'** inside that same folder
8. Open tokenizer_class.py, comment all existing code inside the main loop and run the 'extract_wiki_text' function there.
9. Once text file has been fully written, comment the extract_wiki_text function and uncomment the main loop.
