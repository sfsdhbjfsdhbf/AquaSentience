<template>
  <div class="sidebar">
    <div class="sidebar-header">
      <h2 class="sidebar-title">AI应用中心</h2>
    </div>
    <nav class="sidebar-nav">
      <router-link to="/" class="nav-item" active-class="active">
        <span class="nav-icon">🏠</span>
        <span class="nav-text">首页</span>
      </router-link>
      <router-link to="/manus" class="nav-item" active-class="active">
        <span class="nav-icon">🤖</span>
        <span class="nav-text">AI超级智能体</span>
      </router-link>
      <router-link to="/love" class="nav-item" active-class="active">
        <span class="nav-icon">🖼️</span>
        <span class="nav-text">AI图像水印专家</span>
      </router-link>
      <router-link to="/full-image-protection" class="nav-item" active-class="active">
        <span class="nav-icon">🛡️</span>
        <span class="nav-text">整图内容保护</span>
      </router-link>
      <router-link to="/partial-content-protection" class="nav-item" active-class="active">
        <span class="nav-icon">🧩</span>
        <span class="nav-text">局部内容保护</span>
      </router-link>
    </nav>
    <div class="sidebar-footer">
      <div class="sidebar-footer-content">
        <p class="sidebar-footer-text">
          <span class="sidebar-footer-author">制作者：<span class="author-name">夜未央</span></span>
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
// Sidebar component - no additional logic needed for now
</script>

<style scoped>
.sidebar {
  position: fixed;
  left: 0;
  top: 0;
  width: 280px;
  height: 100vh;
  background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  flex-direction: column;
  box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
  z-index: 1000;
  overflow-y: auto;
}

.sidebar-header {
  padding: 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}

.sidebar-title {
  font-size: 20px;
  font-weight: 600;
  margin: 0;
  text-align: center;
}

.sidebar-nav {
  flex: 1;
  padding: 20px 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 20px;
  color: rgba(255, 255, 255, 0.9);
  text-decoration: none;
  transition: all 0.3s ease;
  border-left: 3px solid transparent;
}

.nav-item:hover {
  background: rgba(255, 255, 255, 0.1);
  color: white;
}

.nav-item.active {
  background: rgba(255, 255, 255, 0.2);
  border-left-color: white;
  color: white;
  font-weight: 600;
}

.nav-icon {
  font-size: 20px;
  width: 24px;
  text-align: center;
}

.nav-text {
  font-size: 14px;
}

.sidebar-footer {
  padding: 20px;
  border-top: 1px solid rgba(255, 255, 255, 0.2);
}

.sidebar-footer-content {
  text-align: center;
}

.sidebar-footer-text {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.8);
  margin: 0;
}

.sidebar-footer-author {
  color: rgba(255, 255, 255, 0.9);
}

.author-name {
  font-weight: 500;
}

/* 移动端样式 */
@media (max-width: 768px) {
  .sidebar {
    width: 100%;
    height: auto;
    max-height: 100vh;
    transform: translateX(-100%);
    transition: transform 0.3s ease;
  }

  .sidebar.open {
    transform: translateX(0);
  }

  .sidebar-header {
    padding: 15px;
  }

  .sidebar-title {
    font-size: 18px;
  }

  .nav-item {
    padding: 14px 20px;
  }

  .nav-text {
    font-size: 15px;
  }
}

/* 滚动条样式 */
.sidebar::-webkit-scrollbar {
  width: 6px;
}

.sidebar::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
}

.sidebar::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.3);
  border-radius: 3px;
}

.sidebar::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.5);
}
</style>
