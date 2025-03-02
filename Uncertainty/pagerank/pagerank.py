import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    probability_distribution = {}
    size_of_corpus = len(corpus)
    links = corpus.get(page)

    # Handle case where there are no outgoing links
    if not links:
        # If no links, each page has an equal probability
        for p in corpus:
            probability_distribution[p] = 1 / size_of_corpus
        return probability_distribution

    # Calculate probabilities
    random_choice = (1 - damping_factor) / size_of_corpus
    link_probability = damping_factor / len(links)
    
    for p in corpus:
        if p in links:
            probability_distribution[p] = random_choice + link_probability
        else:
            probability_distribution[p] = random_choice

    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pagerank_counts = {page: 0 for page in corpus}

    current_page = random.choice(list(corpus.keys()))
    
    for _ in range(n):
        pagerank_counts[current_page] += 1
        transition_probabilities = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(transition_probabilities.keys()), weights=list(transition_probabilities.values()), k=1)[0]

    pagerank = {page: count / n for page, count in pagerank_counts.items()}

    return pagerank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    epsilon=1e-6
    size_of_corpus = len(corpus)
    random_choice = (1 - damping_factor) / size_of_corpus

    pagerank = {page: 1 / size_of_corpus for page in corpus}

    while True:
        new_prp = dict()
        for dest in pagerank:
            sum_ranks = 0
            for source in pagerank:
                links = corpus[source]
                if dest in links:
                    sum_ranks += pagerank[source] / len(links)
                elif not links:
                    sum_ranks += pagerank[source] / size_of_corpus
            
            new_prp[dest] = random_choice + (damping_factor * sum_ranks)

        max_difference = max(abs(new_prp[page] - pagerank[page]) for page in pagerank)
        if max_difference < epsilon:
            break
        
        pagerank = new_prp

    total = sum(pagerank.values())
    pagerank = {page: rank / total for page, rank in pagerank.items()}

    return pagerank

if __name__ == "__main__":
    main()
