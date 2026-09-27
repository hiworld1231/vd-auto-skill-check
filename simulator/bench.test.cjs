const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

function bench(){
  const html=fs.readFileSync(__dirname+'/skillcheck.html','utf8');
  const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
  let now=0,frame,timers=[],keydown;
  const nodes=new Map();
  const document={getElementById(id){
    if(!nodes.has(id)) nodes.set(id,{style:{},textContent:'',value:'60',setAttribute(){},addEventListener(){}});
    return nodes.get(id);
  }};
  const window={};
  vm.runInNewContext(script,{document,window,performance:{now:()=>now},
    addEventListener:(name,cb)=>{if(name==='keydown')keydown=cb},
    requestAnimationFrame:cb=>{frame=cb},
    setTimeout:(cb,delay)=>{const timer={cb,at:now+delay};timers.push(timer);return timer},
    clearTimeout:id=>{timers=timers.filter(timer=>timer!==id)}});
  return {
    state:()=>window.VDBenchState(),
    advance(ms){
      const until=now+ms;
      while(now<until){
        timers.sort((a,b)=>a.at-b.at);
        const nextTimer=timers[0];
        const nextFrame=now+1000/60;
        if(nextTimer&&nextTimer.at<=until&&nextTimer.at<=nextFrame){
          now=nextTimer.at;timers.shift();nextTimer.cb();
        }else if(nextFrame<=until){
          now=nextFrame;if(frame)frame(now);
        }else break;
      }
      now=until;
    },
    key(code){keydown({code,repeat:false,preventDefault(){}})},
    result:()=>document.getElementById('result').textContent,
    setResponseDelay(ms){return window.VDBenchSetResponseDelay(ms)},
  };
}

test('bench starts ordinary checks and Frenzy keeps the rotating phase',()=>{
  const b=bench();
  b.advance(600);
  assert.equal(b.state().count,1);
  const before=b.state().phase;
  b.key('KeyF');
  assert.equal(b.state().count,2);
  assert.equal(b.state().frenzy,true);
  assert.equal(b.state().phase,before);
  b.advance(150);
  assert.ok(b.state().phase>before);
});

test('a miss stops Frenzy and a new ordinary check can start',()=>{
  const b=bench();
  b.key('KeyF');
  assert.equal(b.state().count,1);
  b.key('Space');
  b.advance(60);
  assert.match(b.result(),/MISS/);
  assert.equal(b.state().frenzy,false);
  b.advance(900);
  assert.equal(b.state().count,2);
});

test('fast Frenzy timing includes the configured Space response delay',()=>{
  const b=bench();
  b.key('KeyF');
  const state=b.state();
  const leadMs=60;
  assert.equal(state.responseDelayMs,leadMs);
  const phaseAtPress=state.targetPhase+5-700*leadMs/1000;
  const waitMs=(phaseAtPress-state.phase)/700*1000;

  b.advance(waitMs);
  b.key('Space');
  assert.equal(b.result(),'');
  b.advance(60);

  assert.match(b.result(),/GREAT/);
});

test('Space response delay can match an explicit solver lead',()=>{
  const b=bench();
  assert.equal(b.setResponseDelay(35),true);
  assert.equal(b.state().responseDelayMs,35);
  assert.equal(b.setResponseDelay(301),false);
  assert.equal(b.state().responseDelayMs,35);
});
