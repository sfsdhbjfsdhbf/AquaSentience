package com.ywy.ywyagent.controller;

import com.ywy.ywyagent.app.LoveApp;
import com.ywy.ywyagent.agent.YwyManus;
import com.ywy.ywyagent.context.ChatIdContext;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import reactor.core.publisher.Flux;
import com.ywy.ywyagent.tools.AskHumanTool;
import org.springframework.http.ResponseEntity;
import java.util.Map;
import java.io.IOException;


@Slf4j
@RestController
@RequestMapping("/ai2")
public class AiController {
    @Resource
    private LoveApp loveApp;
    @Resource
    private ToolCallback[] allTools;
    @Resource
    private ChatModel dashscopeChatModel;

    @GetMapping("/love_app/chat/sync")
    public String doChatWithLoveAppSync(String message, String chatId) {
        return loveApp.doChat(message, chatId);
    }

    @GetMapping(value = "/love_app/chat/sse", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> doChatWithLoveAppSSE(String message, String chatId) {
        return loveApp.doChatBySteam(message, chatId);
    }

    /**
     * SSE Emitter 接口 - 支持 AskHumanTool
     *
     * 重要：由于使用 Reactor Flux，ThreadLocal 可能在异步线程中丢失
     * 解决方案：在 subscribe 的回调中设置 ThreadLocal
     */
    @GetMapping("/love_app/chat/sse/emitter")
    public SseEmitter doChatWithLoveAppEmitter(String message, String chatId) {
        SseEmitter emitter = new SseEmitter(180000L);

        // 注册 SSE 连接
        AskHumanTool.registerEmitter(chatId, emitter);

        // 在 subscribe 时设置 ThreadLocal（因为 Reactor 会在不同线程中执行）
        loveApp.doChatBySteam(message, chatId)
                .doOnSubscribe(subscription -> {
                    // 在订阅时设置 ThreadLocal
                    ChatIdContext.set(chatId);
                    log.info("设置 ThreadLocal chatId: {}", chatId);
                })
                .doOnNext(chunk -> {
                    // 在每个数据块处理时也设置一次（确保 ThreadLocal 存在）
                    // 因为 Reactor 可能在不同线程中处理
                    ChatIdContext.set(chatId);
                })
                .subscribe(
                        chunk -> {
                            try {
                                // 确保 ThreadLocal 存在
                                ChatIdContext.set(chatId);
                                emitter.send(chunk);
                            } catch (IOException e) {
                                emitter.completeWithError(e);
                            }
                        },
                        error -> {
                            ChatIdContext.clear();
                            emitter.completeWithError(error);
                        },
                        () -> {
                            ChatIdContext.clear();
                            emitter.complete();
                        }
                );

        return emitter;
    }

    @GetMapping("/manus/chat")
    public SseEmitter DoChatWithManus(String message, String chatId) {
        // 👇 添加 chatId 参数，如果前端没有传递，才生成新的
        if (chatId == null || chatId.isEmpty()) {
            chatId = "manus_" + System.currentTimeMillis();
        }

        YwyManus ywyManus = new YwyManus(allTools, dashscopeChatModel);
        SseEmitter emitter = ywyManus.runStream(message);

        // 注册 SSE 连接
        AskHumanTool.registerEmitter(chatId, emitter);

        return emitter;
    }

    @PostMapping("/askHuman/answer")
    public ResponseEntity<?> receiveAnswer(@RequestBody Map<String, String> request) {
        String answer = request.get("answer");
        String chatId = request.get("chatId");
        log.info(answer+chatId+"safhjsdfhjbsdhjbfshjdbfjh");
        AskHumanTool.receiveAnswer(chatId, answer);

        return ResponseEntity.ok().build();
    }
}









