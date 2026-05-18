import { useEffect, useRef } from 'react';

class Particle {
  x: number = 0;
  y: number = 0;
  vx: number = 0;
  vy: number = 0;
  r: number = 0;
  alpha: number = 0;
  isNode: boolean = false;
  pulse: number = 0;
  canvasWidth: number = 0;
  canvasHeight: number = 0;

  constructor(w: number, h: number) {
    this.canvasWidth = w;
    this.canvasHeight = h;
    this.reset();
  }

  reset() {
    this.x = Math.random() * this.canvasWidth;
    this.y = Math.random() * this.canvasHeight;
    this.vx = (Math.random() - 0.5) * 0.3;
    this.vy = (Math.random() - 0.5) * 0.3;
    this.r = Math.random() * 1.5 + 0.3;
    this.alpha = Math.random() * 0.5 + 0.1;
    this.isNode = Math.random() < 0.03;
    if (this.isNode) {
      this.r = 3;
      this.alpha = 0.9;
      this.pulse = 0;
    }
  }

  update(w: number, h: number) {
    this.canvasWidth = w;
    this.canvasHeight = h;
    this.x += this.vx;
    this.y += this.vy;
    if (this.x < 0 || this.x > this.canvasWidth || this.y < 0 || this.y > this.canvasHeight) {
      this.reset();
    }
    if (this.isNode) {
      this.pulse = (this.pulse + 0.04) % (Math.PI * 2);
    }
  }

  draw(ctx: CanvasRenderingContext2D) {
    if (this.isNode) {
      const glow = Math.sin(this.pulse) * 0.4 + 0.6;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(74, 222, 128, ${this.alpha * glow})`; // using green-400 equivalent
      ctx.fill();
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r + (Math.sin(this.pulse) * 4 + 4), 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(74, 222, 128, ${0.15 * glow})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(52, 211, 153, ${this.alpha * 0.5})`; // emerald-400 slightly dim
      ctx.fill();
    }
  }
}

export const LiveBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let particles: Particle[] = [];
    let scanY = 0;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      // Re-initialize particles on resize if empty
      if (particles.length === 0) {
        for (let i = 0; i < 120; i++) {
          particles.push(new Particle(canvas.width, canvas.height));
        }
      }
    };

    window.addEventListener('resize', resize);
    resize();

    const drawConnections = () => {
      for (let i = 0; i < particles.length; i++) {
        if (!particles[i].isNode) continue;
        for (let j = i + 1; j < particles.length; j++) {
          if (!particles[j].isNode) continue;
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < 250) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(74, 222, 128, ${0.12 * (1 - d / 250)})`;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
        }
      }
    };

    const loop = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Subtle scanline effect sweeping down
      ctx.fillStyle = 'rgba(74, 222, 128, 0.012)';
      ctx.fillRect(0, scanY, canvas.width, 80);
      ctx.fillStyle = 'rgba(74, 222, 128, 0.03)';
      ctx.fillRect(0, scanY, canvas.width, 2);
      scanY = (scanY + 1.2) % canvas.height;

      drawConnections();
      particles.forEach((p) => {
        p.update(canvas.width, canvas.height);
        p.draw(ctx);
      });

      animationFrameId = window.requestAnimationFrame(loop);
    };

    loop();

    return () => {
      window.removeEventListener('resize', resize);
      window.cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 z-0 pointer-events-none opacity-60 mix-blend-screen"
    />
  );
};
