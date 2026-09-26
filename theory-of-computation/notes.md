a proof for the *construction* of a document processing service.

**def.** document processing service

a system $P$ is a document processing service if given raw documents d, the system produces a structured representation $r(d)$ such
that all required information needed for downstream services is preserved.

**def.** required-information

required-information := {id, title, authors, year, text, citations, source_path}

<br>

**construction**

suppose a user has a collection of raw documents $D = \{d_1, d_2, ..., d_n\}$ on their filesystem, and they would like to upload them
to our document processing system,

we must first define how a user communicates (interacts) with our system.

suppose our system provides a **command line user interface** (cli), then the user shall give the system the file path pointing to the 
location of their documents.

once the system has been given the file path, it can begin processing the documents within that file path.

given a file path pointing to the directory of documents the user wants processed, how should we go about processing these documents?

there are two approaches that stand out:

1. process each document *sequentially*
2. process multiple documents *concurrently*

suppose we process each document sequentially,

the following sequence of events must occur,

1. generate document id
2. check if we've generated this id before (duplicate)
3. extract title
4. extract author
5. extract year
6. extract text
7. extract citations
8. extract source_path
9. return Document(title, author, year, text, citations, source_path)
10. save Document to **corpus**

suppose we process documents *concurrently*, what exaclty does this mean?

**concurrency** is the general ability of a system to *make progress* on multiple *tasks* during *overlapping periods of time*

There three primary approaches to concurrency:
- multiprocessing
- mulithreading
- asynchronus programming

concurrency: overlapping progress
parallelism: simultaneous execution

**multiproecessing** uses multiple os *processes*:

process 1 -> task A
process 2 -> task B

and if those processes run on different cpu cores, they may execute literally at the same instant. that is parallelism.

**question**: how to discern whether you need conurrency or parallelism, and how do you choose the appraoch to obtaining it?


**does the document processing system work as expected:**

