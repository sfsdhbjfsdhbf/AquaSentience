package com.ywy.ywyagent.agent;

import lombok.Data;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;

import java.time.Instant;

@Component
public class AgentStuckMonitor {

    private final Sinks.Many<MonitorEvent> sink =
            Sinks.many().multicast().onBackpressureBuffer();

    public void report(MonitorEvent event) {
        sink.tryEmitNext(event);
    }

    public Flux<MonitorEvent> stream() {
        return sink.asFlux();
    }

    @Data
    public static class MonitorEvent {
        private String agentName;
        private String state;
        private int similarTextCount;
        private int noToolCallCount;
        private String lastText;
        private boolean stuck;
        private Instant timestamp;
    }
}