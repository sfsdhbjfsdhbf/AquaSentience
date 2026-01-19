import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import ManusApp from '../views/ManusApp.vue'
import LoveApp from '../views/LoveApp.vue'
import FullImageProtection from '../views/FullImageProtection.vue'
import PartialContentProtection from '../views/PartialContentProtection.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/manus',
    name: 'ManusApp',
    component: ManusApp
  },
  {
    path: '/love',
    name: 'LoveApp',
    component: LoveApp
  },
  {
    path: '/full-image-protection',
    name: 'FullImageProtection',
    component: FullImageProtection
  },
  {
    path: '/partial-content-protection',
    name: 'PartialContentProtection',
    component: PartialContentProtection
  },
  {
    path: '/protection',
    redirect: '/full-image-protection'
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
