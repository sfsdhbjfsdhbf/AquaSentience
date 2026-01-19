package com.ywy.ywyagent.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
@RestController
@RequestMapping("/healthy")
public class HealthyController {
    @GetMapping
    public String Hello(){
    return "ok";
    }
}
