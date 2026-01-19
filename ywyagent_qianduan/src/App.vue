<template>
  <div id="app">
    <Sidebar />
    <div class="main-content">
      <router-view />
    </div>
  </div>
</template>

<script setup>
import Sidebar from './components/Sidebar.vue'
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background-color: #f5f5f5;
}

#app {
  width: 100%;
  min-height: 100vh;
  display: flex;
  position: relative;
}

.main-content {
  flex: 1;
  width: 100%;
  transition: margin-right 0.3s ease;
}

/* PC端：为导航栏留出空间（导航栏在左侧） */
@media (min-width: 769px) {
  .main-content {
    margin-left: 280px;
  }
}

/* 移动端：导航栏覆盖在内容上 */
@media (max-width: 768px) {
  .main-content {
    margin-right: 0;
  }
}
</style>

