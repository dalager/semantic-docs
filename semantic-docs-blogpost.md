# Software Documentation in the AI Era

> How do we deal with the documentation swamp we're slowly sinking into when AI tools readily talk the devil and everyone else's ear off?

(_Translated from my Danish blogpost, October 28, 2025: <https://dalager.com/blog/2025/semantic-docs/>_)

![A girl and a robot stand hand in hand looking at a landscape filled with documentation. There's a somewhat hopeless atmosphere. (AI-gen)](/img/semantic-docs-header.png)

## TL;DR

- AI agents produce a lot of text and are notoriously hesitant to delete things.
- This very much applies to documentation as well.
- The consequence is that it becomes harder to navigate, maintain, and find the right information.
- A partial solution might be to use classical techniques and AI language technology tools against the problem.
- Fight 🔥 with 🔥!

## Documentation and Large Language Models

When it comes to documenting software systems, what constitutes good documentation for a large language model is usually also good documentation for humans.

For an AI model to help you, it needs your unique context.

It knows a lot about the world, but it's a bit lost when you ask it about your own project, and you get something like "Listen, buddy, rewind a bit – I just need to know a little more about your situation."

Context is what gets sent to ChatGPT every single time you hit 'send', and in a chat session, it's all the questions and answers in your current chat thread and your general instructions about how you want to be treated, etc.

When an AI tool needs to say something about your software project, it also needs context.
And often that's documentation in the form of text files in the bare-bones Markdown format and, of course, source code.

The better the map of our software system, the better the AI model should be able to answer questions and solve programming tasks.

The problem is just that it quickly reaches a dead end like the country in the incredibly short Borges novella [On Exactitude in Science](https://en.wikipedia.org/wiki/On_Exactitude_in_Science), where an empire, so obsessed with the noble art of cartography, suffocates itself under a map at 1:1 scale.

## Challenges with Documentation in AI-Boosted Projects

For those of us who make software, documentation is a double-edged sword and always has been.
It's fantastic when it's easy to read, contains the information you need, and is up to date.
Otherwise, it can be worse than nothing.

With AI tools comes more code that developers haven't written themselves. Maybe they haven't even read the code, but know roughly what's going on, and hopefully it's been tested very thoroughly.

If you're conscientious as an AI-powered developer, you also ensure that more documentation is produced continuously — both so you and the AI models can more easily understand what's going on.

But: If there's one thing we've all figured out, it's that language models have what you might call an agency problem: They feel very strongly about language – it's like their identity. So a LOT of language and documentation comes out.

My experience is that a lot of writing happens, not just because the LLM wants it, but because you yourself want it:

- What is this thing, broadly speaking?
- What is this subsystem about, kind of high level?
- What are we using this module in this subsystem for?
- What are we going to build?
- What has been built?
- How should we build things?
- How far have we gotten in our plans?
- How should Claude/Copilot/Cline/Cursor behave?

All things that need to be saved as Markdown files.

Viewed as a static collection of information, it would be challenging to work with.

But as documentation of a living software system in continuous development by AI agents with fresh developer types at the helm, it's a veritable nightmare:

### Problems for Humans and Machines

- Where should we document this?
- What should the documentation look like?
- What level of abstraction should be used where?
- How can my AI agent remember to update the documentation in the right place?
- How do I ensure that outdated documentation and other irrelevant text documents are deleted, and when?

## Strategy: Use Even More AI Tools

We already benefit greatly from a number of tools that these AI models can use:
Automated tests, formatting, type checking, code duplication scanners, complexity metrics, vulnerability scanners, and all sorts of other checks that ensure the code maintains some form of rigor.
And when an AI tool has been at work, these tools are run right after, and no matter how much you've insisted on proper behavior and described how the code is expected to look, there's ALWAYS something: something isn't quite as desired.
Confronted with the problem, the AI model naturally comes with a lot of talking around it and poor excuses and confirmation of my deep insights into my exposure of the sloppiness.
And then it has to get back on the horse.

So. What if you could do the same with documentation?

### Principles

The idea is that we use proven techniques from our high-tech toolbox:

#### Language Technology

- Semantic search with embeddings in a vector database
- Clustering – grouping based on embeddings
- AI summarization
- AI labeling (zero- or few-shot)

#### Modern Software Practice

- Documentation as CI focus: validation, formatting, and analysis of placement, duplicate content, complexity, etc.
- The shift-left principle from the DevOps movement, but extended to documentation, so you think _gardening_ into the processes from the start
- Developer tools that make it hard for developers to fail

## The Experiment, a Case, and a Conceptual Tool

### The Project I Wanted to Save

My main hobby project at the moment is written in Python. It contains about 40,000 lines of code and is split about 50/50 between code and tests.
It consists of a core, an API, a serverless RunPod wrapper, a CI/CD pipeline, an OpenTelemetry stack, OIDC auth, HuggingFace and Cloudflare R2 integrations, and also a hand-rolled load-testing system with some report generation 😅.

In other words: there are many spices, even though it's a fairly simple dish, and I've **really** struggled to keep the documentation in check.

### Away from the Swamp

I've built `semantic-docs`, a small Python tool that maintains a semantic database locally ([ChromaDB](https://docs.trychroma.com/docs/overview/introduction) + SQLite) with all Markdown files in my project.

It can be used both as a tool that's tightly integrated into the ongoing development process and as an analysis tool for existing documentation in projects.

#### Integrated into the Development Process

1. When a Markdown file is updated or added, it's automatically indexed with embeddings, and GPT-5 is used to summarize the content and save a summary and some labels or tags in the index as well.
To weight the placement of each file, the full path, e.g., `/docs/guidelines/testing/tdd_recipe.md`, is also included in the semantic weighting.

1. After that, the documentation file is validated against the rest of the documentation:

- Is there semantic overlap between this and other documentation?
- Is the file placed in a sensible location in the overall documentation structure?

1. Then it's checked by a "documentation agent" that performs a documentation review of the document and makes sure to remove questionable LLM fluff like "You are awesome! And the project is now perfect and ready for production!"

If something is wrong here, the developer and especially the AI tool, such as Claude Code, gets an error with a description of the problem. You can then try to do something about it.

#### Analysis of the Documentation Codebase

Part of the tool is a clustering function that, based on the documentation in the vector database, tries to find semantic groupings across files.

![Clusters](/img/viz09.png)

Combining these patterns with an analysis of where there's a conflict between physical placement and semantic content – that is, when someone has put their underwear in the sock drawer – we're approaching something that addresses part of our needs.

![Folder-cluster analysis](/img/semantic-docs-folderanalysis.png)

It's not entirely easy to understand that image, I think, but if you translate it to a concrete application, it makes sense, I believe.

Here I've run the tool on the entire project:

```text
=================================================================
🏗️  FOLDER-CLUSTER STRUCTURE COMPARISON
=================================================================

📊 Overview:
   Total Folders: 23
   Total Clusters: 6
   Total Documents: 57
   Generated: 2025-08-26T22:06:04.326640

🎯 Overall Alignment:
   🟠 Quality: FAIR (score: 0.464)
   📁 Folder Purity: 0.877 (how focused folders are)
   🗂️  Cluster Homogeneity: 0.316 (how unified clusters are)

📁 Folders Needing Attention (lowest purity first):
   1. 🟡 docs/guides/architecture - purity: 0.60
      (5 docs across 2 clusters)
   2. 🟡 docs/development - purity: 0.67
      (3 docs across 2 clusters)
   3. 🟡 docs/implementation/features - purity: 0.67
      (3 docs across 2 clusters)
   4. 🟡 docs - purity: 0.75
      (4 docs across 2 clusters)
   5. 🟡 docs/guides/processes - purity: 0.75
      (4 docs across 2 clusters)

🗂️  Clusters Needing Attention (lowest homogeneity first):
   1. 🔴 Cluster 2 (root) - homogeneity: 0.15
      (26 docs from 13 folders)
   2. 🔴 Cluster 0 (semantic-docs) - homogeneity: 0.33
      (3 docs from 3 folders)
   3. 🔴 Cluster 1 (tests) - homogeneity: 0.33
      (15 docs from 6 folders)
   4. 🔴 Cluster 4 (docs/architecture/adr) - homogeneity: 0.33
      (3 docs from 3 folders)
   5. 🔴 Cluster 3 (docs/guides/architecture) - homogeneity: 0.40
      (5 docs from 3 folders)
```

And similarly, you can check individual documents for whether they have a "good placement."

## Conclusion

I've used Claude Code as the AI agent platform for the experiment, because Anthropic with their MCP protocol, Sub Agents, Hooks, and Commands are furthest ahead with tooling, I think.

Tooling that helps us create these islands of non-determinism, which are a necessity so that our software solutions don't just become big balls of mud, but muddy plains inhabited by AI ghosts that parents warn their children about venturing into.

### Good

- Really a good process and way to think things through.
- Especially semantic cluster vs. file placement gives a good indication to work with.

### Areas for Improvement

- It's not wildly fast as quick feedback when developing.
- It's a bit cumbersome to get Claude Hooks and Git pre-commit hooks to provide good ergonomics when working.
- The hassle:value ratio is on the edge. But maybe that's because I've spent so long building it that I think so.
- Maybe it's just a slightly over-thought idea.

### Takeaways

Is this something I'll continue working on?
It's certainly something I'll continue thinking about, and it will definitely influence the way I think about documentation in the future, but whether I'll do something in this style, that's a bit unclear.

This concrete tool here might end up in the pile of prototypes that made me smarter, but not the world better 🤷🐰

NB: Human readers of documentation are probably a dying breed. And it's not entirely without irony, but that's a bit of a different conversation.

## References

**ThoughtWorks Technology Podcast: [Caring about documentation in the LLM era](https://sites.libsyn.com/130291/caring-about-documentation-in-the-llm-era-w-heidi-waterhouse)**

Experienced tech writer in the studio on the subject.

**Latent Space Podcast: [Long Live Context Engineering - with Jeff Huber of Chroma](https://www.youtube.com/watch?v=pIbIZ_Bxl_g)**

Very exciting episode about context problems and solutions in LLMs.
