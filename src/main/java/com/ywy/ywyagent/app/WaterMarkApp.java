package com.ywy.ywyagent.app;

import com.ywy.ywyagent.advisor.MyLoggerAdvisor;
import com.ywy.ywyagent.chatMemory.FileBasedChatMemory;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.QuestionAnswerAdvisor;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;

import java.util.List;

import static org.springframework.ai.chat.client.advisor.AbstractChatMemoryAdvisor.CHAT_MEMORY_CONVERSATION_ID_KEY;
import static org.springframework.ai.chat.client.advisor.AbstractChatMemoryAdvisor.CHAT_MEMORY_RETRIEVE_SIZE_KEY;
@Component
@Slf4j
public class WaterMarkApp {
    String fileDir = System.getProperty("user.dir")+"/chat-memory";

    record WaterMarkReport(String title, List<String> summary){

    }
    private final ChatClient chatClient;
    private static final  String  SYSTEM_PROMPT = """
            你是一名图像水印与数字内容保护工程专家。你可以利用 RAG 检索（系统内置的整图水印与对象级水印知识）以及各种工具，为用户提供可靠、准确的专业建议。你的核心职责是根据用户的保护目标，为其提供适合的图像水印方案，包括整图保护和局部区域保护。
            
            你主要处理两类需求：
         
            一、整图内容保护
            典型需求包括：保护整张图像的内容。
            当用户涉及整图保护时，应从 RAG 中检索整体水印方向的相关知识，根据检索结果为用户提供适合的方案或建议。
            
            二、局部内容保护
            典型需求包括：保护图像中的某个局部区域或特定对象。
            当用户涉及对象级保护时，应从 RAG 中检索对象级水印方向的相关知识，根据检索结果为用户提供相应的方案或建议。
            
            如果用户未明确属于整图保护或局部保护，必须主动询问用户：“你要实现整图内容保护还是局部内容保护？”在明确需求后继续回答。
            
            对话策略如下：
            
            在回答前先判断用户需求类型，如不明确需主动澄清。
            
            根据判断方向优先使用 RAG 检索相关知识，如果用户的问题无法再RAG中找到使用网络搜索工具补充。
            
            当回答涉及技术原理、方法、机制等专业内容时，不得虚构不存在的技术或结论，应基于 RAG 或可检索信息作答。
            
            当用户明确要求生成报告、总结、PDF、Markdown、文稿或其他特定格式内容时，应根据用户要求自由组织内容，不受检索规则限制,按Agent User的对话形式保存。
            
            内容必须准确、清晰，如用户的问题不属于整图保护或局部保护范围，应说明该需求不在系统处理范围内。
            """;
    public WaterMarkApp(ChatModel dashscopeChatModel){
        String fileDir = System.getProperty("user.dir") + "/chat-memory";

//        ChatMemory chatMemory = new InMemoryChatMemory();
        ChatMemory chatMemory = new FileBasedChatMemory(fileDir);
        chatClient = ChatClient.builder(dashscopeChatModel)
                .defaultSystem(SYSTEM_PROMPT)
                .defaultAdvisors(
                        new MessageChatMemoryAdvisor(chatMemory)
//                        new Re2Advisor(),
//                        new MyLoggerAdvisor()
                )
                .build();
    }
    public String doChat(String message,String chatId){
        ChatResponse response = chatClient
                .prompt()
                .user(message)
                .advisors(spec -> spec.param(CHAT_MEMORY_CONVERSATION_ID_KEY,chatId)
                        .param(CHAT_MEMORY_RETRIEVE_SIZE_KEY,10))
                .call()
                .chatResponse();
        String content = response.getResult().getOutput().getText();
        log.info("content:{}",content);
        return content;

    }
    public WaterMarkApp.WaterMarkReport doChatWithReport(String message, String chatId){
        WaterMarkApp.WaterMarkReport report = chatClient
                .prompt()
                .system(SYSTEM_PROMPT + "每次对话后都要生成水印结果，标题为{用户名}的水印报告，内容为建议列表")
                .user(message)
                .advisors(spec -> spec.param(CHAT_MEMORY_CONVERSATION_ID_KEY,chatId)
                        .param(CHAT_MEMORY_RETRIEVE_SIZE_KEY,10))
                .call()
                .entity(WaterMarkApp.WaterMarkReport.class);

        log.info("Watermark report:{}",report);
        return report;

    }
    @Resource
    private VectorStore watermarkAppVectorStore;
    public String doChatWithRag(String message,String chatId){
        ChatResponse chatResponse = chatClient.prompt()
                .user(message)
                .advisors((spec -> spec.param(CHAT_MEMORY_CONVERSATION_ID_KEY, chatId)
                        .param(CHAT_MEMORY_RETRIEVE_SIZE_KEY, 10)))
                .advisors(new MyLoggerAdvisor())
                .advisors(new QuestionAnswerAdvisor(watermarkAppVectorStore))
                .call()
                .chatResponse();
        String content = chatResponse.getResult().getOutput().getText();
        log.info("content:{}",content);
        return content;
    }

    @Resource
    private Advisor loveAppRagCloudAdvisor;
    public String doChatWithRag2(String message, String chatId) {
        ChatResponse chatResponse = chatClient
                .prompt()
                .user(message)
                .advisors(spec -> spec.param(CHAT_MEMORY_CONVERSATION_ID_KEY, chatId)
                        .param(CHAT_MEMORY_RETRIEVE_SIZE_KEY, 10))

                .advisors(new MyLoggerAdvisor())

                .advisors(loveAppRagCloudAdvisor)
                .call()
                .chatResponse();
        String content = chatResponse.getResult().getOutput().getText();
        log.info("content: {}", content);
        return content;
    }

    @Resource
    private ToolCallback[] allTools;
    public String doChatWithTools(String message,String chatId){
        ChatResponse response =chatClient.prompt()
                .user(message)
                .advisors(spec -> spec.param(CHAT_MEMORY_CONVERSATION_ID_KEY, chatId)
                        .param(CHAT_MEMORY_RETRIEVE_SIZE_KEY, 10))
                .advisors(new MyLoggerAdvisor())
                .tools(allTools)
                .call()
                .chatResponse();
        String content = response.getResult().getOutput().getText();
        log.info("content: {}", content);
        return content;
    }
    @Resource
    private ToolCallbackProvider toolCallbackProvider;
    public String doChatWithMcp(String message,String chatId){
        ChatResponse response = chatClient.prompt()
                .user(message)
                .advisors(spec -> spec.param(CHAT_MEMORY_CONVERSATION_ID_KEY, chatId)
                        .param(CHAT_MEMORY_RETRIEVE_SIZE_KEY, 10))
                .advisors(new MyLoggerAdvisor())
                .tools(toolCallbackProvider)
                .call()
                .chatResponse();
        String content = response.getResult().getOutput().getText();
        log.info("content:{}",content);
        return content;
    }
    public Flux<String> doChatBySteam(String message, String chatId){
        return chatClient.prompt()
                .user(message)
                .advisors(spec -> spec.param(CHAT_MEMORY_CONVERSATION_ID_KEY,chatId)
                        .param(CHAT_MEMORY_RETRIEVE_SIZE_KEY,10))
                .stream()
                .content();
    }
    public String doChatWithRagAndTools(String message, String chatId) {
        ChatResponse response = chatClient
                .prompt()
                .user(message)
                .advisors(spec -> spec
                        .param(CHAT_MEMORY_CONVERSATION_ID_KEY, chatId)
                        .param(CHAT_MEMORY_RETRIEVE_SIZE_KEY, 10))
                .advisors(new MyLoggerAdvisor())
                .advisors(new QuestionAnswerAdvisor(watermarkAppVectorStore))  // RAG
                .tools(allTools)                                          // 全部 Tools
                .call()
                .chatResponse();

        String content = response.getResult().getOutput().getText();
        log.info("content: {}", content);
        return content;
    }
    public Flux<String> doChatWithRagAndToolsStream(String message, String chatId) {


        log.info("Streaming chat start. chatId={}, userMessage={}", chatId, message);

        return chatClient
                .prompt()
                .user(message)
                .advisors(spec -> spec
                        .param(CHAT_MEMORY_CONVERSATION_ID_KEY, chatId)
                        .param(CHAT_MEMORY_RETRIEVE_SIZE_KEY, 10))
                .advisors(new MyLoggerAdvisor())
                .advisors(new QuestionAnswerAdvisor(watermarkAppVectorStore))  // RAG
                .tools(allTools)                                          // 全部 Tools
                .stream()
                .content();
    }

}
