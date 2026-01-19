package com.ywy.ywyagent.agent;

import cn.hutool.core.collection.CollUtil;
import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatOptions;
import com.ywy.ywyagent.agent.model.AgentState;
import com.ywy.ywyagent.agent.model.TfidfSimilarity;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.messages.*;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.ToolExecutionResult;
import org.springframework.ai.tool.ToolCallback;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@EqualsAndHashCode(callSuper = true)
@Data
@Slf4j
public class ToolCallAgent extends ReActAgent {

    private final ToolCallback[] availableTools;
    private final ToolCallingManager toolCallingManager;
    private final ChatOptions chatOptions;

    private ChatResponse toolCallChatResponse;

    // -----------------------------
    // STUCK 状态
    // -----------------------------
    private String lastAssistantText = null;
    private int similarTextCount = 0;
    private int noToolCallCount = 0;


    private static final int NO_TOOL_THRESHOLD = 3;        // 连续 3 次未调用工具


    public ToolCallAgent(ToolCallback[] availableTools) {
        super();
        this.availableTools = availableTools;

        this.toolCallingManager = ToolCallingManager.builder().build();

        this.chatOptions = DashScopeChatOptions.builder()
                .withProxyToolCalls(true)
                .build();
    }

    // -----------------------------------------------------
    // STUCK 检测（仅文本相似 + 不调用工具）
    // -----------------------------------------------------
    private boolean detectStuck(String responseText, boolean toolCalledThisRound) {



        // 保存本次文本
        lastAssistantText = responseText;

        // -----------------------------------------------------
        // 未调用工具计数
        // -----------------------------------------------------
        if (!toolCalledThisRound) {
            noToolCallCount++;
        } else {
            noToolCallCount = 0;
        }

        // -----------------------------------------------------
        // 判断是否 STUCK
        // -----------------------------------------------------
        if (noToolCallCount >= NO_TOOL_THRESHOLD) {

            log.error("\n\n🔥🔥🔥 [{}] Agent STUCK detected!\n" +
                            "相似输出次数：{}\n" +
                            "未调用工具次数：{}\n" +
                            "最后输出：{}\n",
                    getName(), similarTextCount, noToolCallCount, responseText);

            return true;
        }

        return false;
    }


    // -----------------------------------------------------
    // THINK：模型生成下一步动作（text or tool）
    // -----------------------------------------------------
    @Override
    public boolean think() {

        // 有 nextStepPrompt，则加入消息
        if (getNextStepPrompt() != null && !getNextStepPrompt().isEmpty()) {
            UserMessage userMessage = new UserMessage(getNextStepPrompt());
            getMessageList().add(userMessage);
        }

        Prompt prompt = new Prompt(getMessageList(), chatOptions);

        try {
            ChatResponse chatResponse = getChatClient()
                    .prompt(prompt)
                    .system(getSystemPrompt())
                    .tools(availableTools)
                    .call()
                    .chatResponse();

            this.toolCallChatResponse = chatResponse;

            AssistantMessage assistantMessage = chatResponse.getResult().getOutput();
            String resultText = assistantMessage.getText();
            List<AssistantMessage.ToolCall> toolCalls = assistantMessage.getToolCalls();

            log.info("🤖 [{}] THINK 输出文本：{}", getName(), resultText);
            log.info("🤖 [{}] 检测到工具调用数量：{}", getName(), toolCalls.size());

            if (!toolCalls.isEmpty()) {
                String toolInfo = toolCalls.stream()
                        .map(t -> "🛠 工具：" + t.name() + " | 参数：" + t.arguments())
                        .collect(Collectors.joining("\n"));
                log.info("{}", toolInfo);
            }

            boolean toolCalledThisRound = !toolCalls.isEmpty();

            // -----------------------------------------------------
            // 调用 STUCK 检测（文本相似 + 未调用工具）
            // -----------------------------------------------------
            // -------------------------------------------------
            if (detectStuck(resultText, toolCalledThisRound)) {

                log.warn("⚠️ [{}] Agent STUCK → 触发 askHuman 工具调用", getName());

                // 1️⃣ 构造 ToolCall（4 参数构造器）
                AssistantMessage.ToolCall askHumanCall =
                        new AssistantMessage.ToolCall(
                                "ask-human-1",      // id
                                "function",         // type（固定）
                                "askHuman",         // 工具名
                                "{\"question\":\"我现在无法继续推理或选择合适的工具，请你告诉我下一步该怎么做？\"}"
                        );

                // 2️⃣ 把 toolCalls 放进 metadata（关键点）
                Map<String, Object> metadata = new HashMap<>();
                metadata.put("tool_calls", List.of(askHumanCall));

                // 3️⃣ 使用「content + metadata」构造 AssistantMessage
                AssistantMessage askHumanMessage =
                        new AssistantMessage(
                                "Agent is stuck and requires human assistance.",
                                metadata
                        );

                // 4️⃣ 写入消息历史
                getMessageList().add(askHumanMessage);

                // 5️⃣ 返回 true → BaseAgent 进入 ACT()
                return true;
            }

            // 没工具 → 输出文本
            if (toolCalls.isEmpty()) {
                getMessageList().add(assistantMessage);
                return false;
            }

            // 有工具 → 执行 act()
            return true;

        } catch (Exception e) {
            log.error("❌ [{}] THINK 异常：{}", getName(), e.getMessage());
            getMessageList().add(new AssistantMessage("模型处理异常：" + e.getMessage()));
            return false;
        }
    }


    // -----------------------------------------------------
    // ACT：执行工具
    // -----------------------------------------------------
    @Override
    public String act() {

        if (!toolCallChatResponse.hasToolCalls()) {
            return "没有工具调用";
        }

        Prompt prompt = new Prompt(getMessageList(), chatOptions);

        ToolExecutionResult toolExecutionResult =
                toolCallingManager.executeToolCalls(prompt, toolCallChatResponse);

        setMessageList(toolExecutionResult.conversationHistory());

        ToolResponseMessage toolResponseMessage =
                (ToolResponseMessage) CollUtil.getLast(toolExecutionResult.conversationHistory());

        String results = toolResponseMessage.getResponses().stream()
                .map(resp -> "🛠 工具 " + resp.name() + " 执行完成 → " + resp.responseData())
                .collect(Collectors.joining("\n"));

        log.info("🔧 [{}] 工具执行结果：\n{}", getName(), results);

        // detect doTerminate
        boolean terminateCalled = toolResponseMessage.getResponses().stream()
                .anyMatch(resp -> "doTerminate".equals(resp.name()));

        if (terminateCalled) {
            log.info("🏁 [{}] 终止工具被调用，Agent 完成", getName());
            setState(AgentState.FINISHED);
        }

        return results;
    }

    private final TfidfSimilarity tfidf = new TfidfSimilarity();
    // -----------------------------------------------------
    // 简单字符 Jaccard 相似度（稳定够用）
    // -----------------------------------------------------
    private boolean textSimilarity(String a, String b) {




        double sim = tfidf.similarity(a, b);


        boolean similarText = sim >= 0.80;  // 推荐阈值 0.75-0.85
        return  similarText;

    }
}
