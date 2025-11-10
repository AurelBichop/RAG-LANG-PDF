import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import express from 'express';

const app = express();
const port = 8989;

// Add CORS headers
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type');
    if (req.method === 'OPTIONS') {
        return res.sendStatus(200);
    }
    next();
});

app.use(express.static('.'));
app.use(express.json());

const llm = new Ollama({
    model: "llama3.2",
    baseUrl: "http://172.17.0.1:11434",
});

const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Answer the user's question based on the following context: {context}."],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
]);

const chain = await createStuffDocumentsChain({
    prompt,
    llm
});


const pdfLoader = new PDFLoader("./pdf-document/renseignements.pdf");
const pdfDocs = await pdfLoader.load();

/* Load all PDFs within the specified directory */
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 250,
    chunkOverlap: 50,
});


//splittedDocs.push(...pdfDocs);
const splittedDocs = await splitter.splitDocuments(pdfDocs);

const ollamaEmbeddings = new OllamaEmbeddings({
    model: "llama3.2",
    baseUrl: "http://172.17.0.1:11434",
});

const vectorStore = await MemoryVectorStore.fromDocuments(splittedDocs, ollamaEmbeddings);

const retriever = vectorStore.asRetriever({
    k: 2,
});

const rephrasePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    ["user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation."],
]);

const historyAwareRetriever = await createHistoryAwareRetriever({
    retriever,
    chatHistorySize: 2,
    llm,
    rephrasePrompt,
});

const conversationChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: historyAwareRetriever,
});

let chatHistory = [];

app.post('/chat', async (req, res) => {
    const userMessage = req.body.message;
    
    chatHistory.push(new HumanMessage(userMessage));
    
    const response = await conversationChain.invoke({
        input: userMessage,
        chat_history: chatHistory,
    });
    
    chatHistory.push(new AIMessage(response.answer));
    
    res.json({ response: response.answer });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});

